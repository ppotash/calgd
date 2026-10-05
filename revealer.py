"""Information-gain revealer for masked diffusion sampling.

The revealer is a small head that scores masked positions by how much
revealing them helps predict the rest of the sequence. It reads the frozen
denoiser's final hidden states plus a few per-token features, and plugs into
`sampling.predictor=ordered sampling.order=revealer`.

Pipeline (denoiser frozen throughout):
  1. mode=revealer_label: for streamed OpenWebText chunks, mask at a random t,
     pick k masked candidates, reveal each (sampled value) and measure the drop
     in mean CE over the remaining masked tokens. Store the gains plus the head's
     inputs at the candidate positions, so training needs no denoiser passes.
  2. mode=revealer_train: train the head with a pairwise ranking loss on the
     stored candidates; report ranking accuracy against confidence baselines.
"""
import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

N_FEATS = 4


class HiddenCapture:
  """Records the input to the backbone's `output_layer` (final hidden states)."""

  def __init__(self, backbone):
    mods = [(n, m) for n, m in backbone.named_modules()
            if n.split('.')[-1] == 'output_layer']
    if len(mods) != 1:
      tail = [n for n, _ in backbone.named_modules()][-20:]
      raise RuntimeError(f'Expected exactly one output_layer, found '
                         f'{[n for n, _ in mods]}. Last modules: {tail}')
    self.name = mods[0][0]
    self.hidden = None
    self.handle = mods[0][1].register_forward_pre_hook(
      self._hook, with_kwargs=True)

  def _hook(self, module, args, kwargs):
    self.hidden = args[0] if args else next(iter(kwargs.values()))

  def remove(self):
    self.handle.remove()


def token_features(log_p, t):
  """Per-position features from the denoiser's log-probs [B, L, V] at time t [B].

  Returns [B, L, N_FEATS]: entropy/5, max prob, log(max prob)/10, t.
  """
  p = log_p.exp()
  ent = -(p * log_p).sum(-1)
  maxp = p.max(-1).values
  tt = t.reshape(-1, 1).to(maxp.dtype).expand_as(maxp)
  return torch.stack(
    [ent / 5.0, maxp, maxp.clamp_min(1e-12).log() / 10.0, tt], dim=-1)


class RevealHead(nn.Module):
  """Per-position MLP: (hidden state, token features) -> reveal score.

  base: None, 'cand' or 'maxp'. With a base, the score is
      log(baseline confidence) + MLP correction,
  where the baseline is the probability of the sampled token ('cand', what
  confidence ordering uses) or the max probability ('maxp'). The last layer
  starts at zero, so an untrained head ranks exactly like the baseline and
  training can only add signal the baseline lacks.
  use_hidden=False ignores the hidden states (features-only ablation).
  """

  def __init__(self, hidden_size, n_feats=N_FEATS, width=128, dropout=0.0,
               base=None, use_hidden=True):
    super().__init__()
    assert base in (None, 'cand', 'maxp'), base
    self.hidden_size, self.n_feats, self.width = hidden_size, n_feats, width
    self.dropout, self.base, self.use_hidden = dropout, base, use_hidden
    self.norm = nn.LayerNorm(hidden_size)
    d_in = (hidden_size if use_hidden else 0) + n_feats
    self.net = nn.Sequential(
      nn.Linear(d_in, width), nn.GELU(), nn.Dropout(dropout),
      nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout),
      nn.Linear(width, 1))
    if base is not None:
      nn.init.zeros_(self.net[-1].weight)
      nn.init.zeros_(self.net[-1].bias)

  def forward(self, hidden, feats, base_score=None):
    parts = [feats.float()]
    if self.use_hidden:
      parts.insert(0, self.norm(hidden.float()))
    out = self.net(torch.cat(parts, dim=-1)).squeeze(-1)
    if self.base is not None:
      assert base_score is not None, f'head expects a {self.base} base score'
      out = out + base_score.float()
    return out

  def save(self, path, extra=None):
    torch.save({'state_dict': self.state_dict(),
                'hidden_size': self.hidden_size, 'n_feats': self.n_feats,
                'width': self.width, 'dropout': self.dropout,
                'base': self.base, 'use_hidden': self.use_hidden,
                'extra': extra or {}}, path)

  @classmethod
  def load(cls, path, map_location='cpu'):
    ck = torch.load(path, map_location=map_location)
    head = cls(ck['hidden_size'], ck['n_feats'], ck['width'],
               ck.get('dropout', 0.0), ck.get('base'),
               ck.get('use_hidden', True))
    head.load_state_dict(ck['state_dict'])
    return head


def base_score(base, feats, cand_prob):
  """Baseline log-score for a residual head ('cand': sampled-token prob)."""
  if base is None:
    return None
  if base == 'cand':
    return cand_prob.float().clamp_min(1e-30).log()
  return feats[..., 1].float().clamp_min(1e-12).log()   # 'maxp'


def _sample_rows(probs):
  """Categorical sample per row of [N, V] in float64 (matches sampling.fp64)."""
  return torch.multinomial(probs.double(), 1).squeeze(-1)


@torch.no_grad()
def info_gain_targets(logprob_fn, x0, x_t, t, log_p0, mask_index, k=8,
                      use_ground_truth=False, chunk=8, n_samples=1):
  """Counterfactual reveal labels.

  For each sequence with more than k masked tokens, pick k masked candidates
  and reveal each one n_samples times with independently sampled values (or
  once with the true value if use_ground_truth), measuring
      gain = mean CE over remaining masked tokens (excluding the candidate),
             before minus after the reveal.
  log_p0: the model's log-probs on x_t, already computed.

  Returns None or dict(seq_idx [S], cand [S,k], vals [S,k,m], gain [S,k,m]).
  """
  m = 1 if use_ground_truth else n_samples
  masked = x_t == mask_index
  seq_idx = (masked.sum(1) > k).nonzero(as_tuple=True)[0]
  if seq_idx.numel() == 0:
    return None
  S, L = seq_idx.numel(), x0.shape[1]
  xs, x0s, ts, ms = x_t[seq_idx], x0[seq_idx], t[seq_idx], masked[seq_idx]
  lp0 = log_p0[seq_idx]

  r = torch.rand(S, L, device=x0.device).masked_fill(~ms, -1.0)
  cand = r.topk(k, dim=1).indices                                    # [S,k]
  if use_ground_truth:
    vals = x0s.gather(1, cand).unsqueeze(-1)                         # [S,k,1]
  else:
    V = lp0.size(-1)
    cl = lp0.gather(1, cand.unsqueeze(-1).expand(-1, -1, V))         # [S,k,V]
    probs = cl.float().exp().view(-1, V).double()
    vals = torch.multinomial(probs, m, replacement=True).view(S, k, m)

  ce0 = -lp0.gather(-1, x0s.unsqueeze(-1)).squeeze(-1).float()       # [S,L]

  # counterfactual inputs, ordered (sequence, candidate, sample) -> [S*k*m, L]
  xc = xs[:, None, None, :].repeat(1, k, m, 1)
  pos = cand[:, :, None, None].expand(-1, -1, m, 1)
  xc.scatter_(3, pos, vals.unsqueeze(-1))
  n = S * k * m
  xc, tc = xc.view(n, L), ts.repeat_interleave(k * m)
  tgt = x0s.repeat_interleave(k * m, dim=0)
  ce_c = []
  for i in range(0, n, chunk):
    lp = logprob_fn(xc[i:i + chunk], tc[i:i + chunk])
    ce_c.append(-lp.gather(-1, tgt[i:i + chunk].unsqueeze(-1))
                .squeeze(-1).float())
  ce_c = torch.cat(ce_c).view(S, k, m, L)

  rem = ms[:, None, :].repeat(1, k, 1)
  rem.scatter_(2, cand.unsqueeze(-1), False)
  rem = rem.float().unsqueeze(2)                                     # [S,k,1,L]
  gain = (((ce0[:, None, None, :] - ce_c) * rem).sum(-1)
          / rem.sum(-1).clamp(min=1))                                # [S,k,m]
  return {'seq_idx': seq_idx, 'cand': cand, 'vals': vals, 'gain': gain}


@torch.no_grad()
def label_batch(model, capture, x0, eps, k, use_ground_truth, chunk,
                n_samples=1):
  """Mask x0 at random t, label k candidates per sequence, and gather the
  head's inputs at the candidates. `model` is a Diffusion module (or anything
  with forward(x, sigma) -> log-probs, noise(t), mask_index)."""
  B, L = x0.shape
  t = eps + (1 - eps) * torch.rand(B, device=x0.device)
  x_t = torch.where(torch.rand(B, L, device=x0.device) < t[:, None],
                    model.mask_index, x0)
  logprob_fn = lambda x, tt: model.forward(x, model.noise(tt)[0])
  log_p0 = logprob_fn(x_t, t)
  hidden = capture.hidden.detach()
  out = info_gain_targets(logprob_fn, x0, x_t, t, log_p0, model.mask_index,
                          k=k, use_ground_truth=use_ground_truth, chunk=chunk,
                          n_samples=n_samples)
  if out is None:
    return None
  si, cand = out['seq_idx'], out['cand']
  gather = lambda a: a[si].gather(
    1, cand.unsqueeze(-1).expand(-1, -1, a.size(-1)))
  feats = gather(token_features(log_p0, t))                          # [S,k,F]
  hid = gather(hidden)                                               # [S,k,H]
  V = log_p0.size(-1)
  lp_c = log_p0[si].gather(1, cand.unsqueeze(-1).expand(-1, -1, V))  # [S,k,V]
  cand_logp = lp_c.gather(-1, out['vals'])                           # [S,k,m]
  return {
    'x0': x0[si].int().cpu(), 'mask': (x_t[si] == model.mask_index).cpu(),
    't': t[si].float().cpu(), 'cand': cand.int().cpu(),
    'gain': out['gain'].float().cpu(), 'feats': feats.float().cpu(),
    'hidden': hid.half().cpu(),
    'cand_prob': cand_logp.float().exp().cpu()}  # prob of each revealed value


def load_shards(files):
  shards = [torch.load(f) for f in files]
  data = {key: torch.cat([s[key] for s in shards]) for key in shards[0]}
  for key in ('gain', 'cand_prob'):   # v1 shards: [S,k] -> [S,k,1]
    if data[key].ndim == 2:
      data[key] = data[key].unsqueeze(-1)
  return data


def pairwise_rank_loss(score, gain, margin=0.0):
  """score, gain: [N, k]. Logistic loss over candidate pairs whose gains differ
  by more than margin: the higher-gain candidate should score higher."""
  ds = score.unsqueeze(2) - score.unsqueeze(1)
  dg = gain.unsqueeze(2) - gain.unsqueeze(1)
  valid = dg.abs() > margin
  if not valid.any():
    return score.sum() * 0.0
  return F.binary_cross_entropy_with_logits(ds[valid], (dg[valid] > 0).float())


@torch.no_grad()
def ranking_metrics(score, gain, margin=0.0):
  """Pairwise accuracy (0.5 = random) and top-1 accuracy (1/k = random)."""
  ds = score.unsqueeze(2) - score.unsqueeze(1)
  dg = gain.unsqueeze(2) - gain.unsqueeze(1)
  valid = dg.abs() > margin
  pair = ((ds > 0) == (dg > 0))[valid].float().mean().item()
  top1 = (score.argmax(1) == gain.argmax(1)).float().mean().item()
  return pair, top1


@torch.no_grad()
def pair_counts(score, gain, margin=0.0):
  """Per-row (correct, valid) pair counts for bootstrapping pair accuracy."""
  ds = score.unsqueeze(2) - score.unsqueeze(1)
  dg = gain.unsqueeze(2) - gain.unsqueeze(1)
  valid = dg.abs() > margin
  correct = (((ds > 0) == (dg > 0)) & valid).sum((1, 2)).float()
  return correct, valid.sum((1, 2)).float()


@torch.no_grad()
def bootstrap_diff(correct_a, correct_b, n_valid, seq_of_row, iters=1000,
                   seed=0):
  """95% interval for pair-acc(a) - pair-acc(b), resampling whole sequences."""
  n_seq = int(seq_of_row.max()) + 1
  agg = lambda v: torch.zeros(n_seq).index_add_(0, seq_of_row, v)
  ca, cb, nv = agg(correct_a), agg(correct_b), agg(n_valid)
  g = torch.Generator().manual_seed(seed)
  idx = torch.randint(0, n_seq, (iters, n_seq), generator=g)
  diffs = (ca[idx].sum(1) - cb[idx].sum(1)) / nv[idx].sum(1).clamp(min=1)
  lo, hi = torch.quantile(diffs, torch.tensor([0.025, 0.975]))
  return float(lo), float(hi)


def group_view(data, target):
  """Ranking groups for training/eval.

  target='sampled': one group per (sequence, sample) - gains and sampled-token
    probs of each candidate under that sample (what the sampler sees).
  target='expected': one group per sequence - gains averaged over samples.
  Returns dict(seq [G], gain [G,k], cand_prob [G,k], t [G]).
  """
  S, k, m = data['gain'].shape
  if target == 'expected':
    return {'seq': torch.arange(S), 'gain': data['gain'].mean(-1),
            'cand_prob': data['cand_prob'].mean(-1), 't': data['t']}
  assert target == 'sampled', target
  seq = torch.arange(S).repeat_interleave(m)
  j = torch.arange(m).repeat(S)
  return {'seq': seq, 'gain': data['gain'][seq, :, j],
          'cand_prob': data['cand_prob'][seq, :, j], 't': data['t'][seq]}


@torch.no_grad()
def label_reliability(gain):
  """Agreement between two halves of the samples, gain [S,k,m] with m >= 2.

  Returns (pair acc of one half's ranking against the other's,
           within-sequence correlation r, Spearman-Brown r for all m samples).
  """
  h = gain.shape[-1] // 2
  g1, g2 = gain[..., :h].mean(-1), gain[..., h:2 * h].mean(-1)
  pair = ranking_metrics(g1, g2)[0]
  c1 = g1 - g1.mean(1, keepdim=True)
  c2 = g2 - g2.mean(1, keepdim=True)
  r = float((c1 * c2).sum() / (c1.pow(2).sum() * c2.pow(2).sum()).sqrt())
  r_full = 2 * r / (1 + r) if r > -1 else float('nan')
  return pair, r, r_full


def owt_chunks(tokenizer, seq_len, skip=0, shuffle_seed=None,
               shuffle_buffer=10_000):
  """Stream OpenWebText as MDLM-style wrapped chunks: [BOS] + text + [EOS],
  documents joined by EOS (BOS = EOS for GPT-2). Yields lists of token ids.
  shuffle_seed: shuffle shard order and documents (deterministic per seed), so
  a short run sees a broad sample of the corpus instead of its first files."""
  import datasets
  ds = datasets.load_dataset('Skylion007/openwebtext', split='train',
                             streaming=True)
  if shuffle_seed is not None:
    ds = ds.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer)
  eos = tokenizer.eos_token_id
  bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else eos
  body = seq_len - 2
  buf, n = [], 0
  for ex in ds:
    buf += tokenizer(ex['text'])['input_ids'] + [eos]
    while len(buf) >= body:
      if n >= skip:
        yield [bos] + buf[:body] + [eos]
      n += 1
      buf = buf[body:]


def shard_files(label_dir):
  return sorted(glob.glob(os.path.join(label_dir, 'shard_*.pt')))
