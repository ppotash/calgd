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
  """Per-position MLP: (hidden state, token features) -> reveal score."""

  def __init__(self, hidden_size, n_feats=N_FEATS, width=256):
    super().__init__()
    self.hidden_size, self.n_feats, self.width = hidden_size, n_feats, width
    self.norm = nn.LayerNorm(hidden_size)
    self.net = nn.Sequential(
      nn.Linear(hidden_size + n_feats, width), nn.GELU(),
      nn.Linear(width, width), nn.GELU(),
      nn.Linear(width, 1))

  def forward(self, hidden, feats):
    h = torch.cat([self.norm(hidden.float()), feats.float()], dim=-1)
    return self.net(h).squeeze(-1)

  def save(self, path, extra=None):
    torch.save({'state_dict': self.state_dict(),
                'hidden_size': self.hidden_size, 'n_feats': self.n_feats,
                'width': self.width, 'extra': extra or {}}, path)

  @classmethod
  def load(cls, path, map_location='cpu'):
    ck = torch.load(path, map_location=map_location)
    head = cls(ck['hidden_size'], ck['n_feats'], ck['width'])
    head.load_state_dict(ck['state_dict'])
    return head


def _sample_rows(probs):
  """Categorical sample per row of [N, V] in float64 (matches sampling.fp64)."""
  return torch.multinomial(probs.double(), 1).squeeze(-1)


@torch.no_grad()
def info_gain_targets(logprob_fn, x0, x_t, t, log_p0, mask_index, k=8,
                      use_ground_truth=False, chunk=8):
  """Counterfactual reveal labels.

  For each sequence with more than k masked tokens, pick k masked candidates,
  reveal each one, and measure
      gain_j = mean CE over remaining masked tokens (excluding j), before - after.
  Revealed values are sampled from the model (as at inference) unless
  use_ground_truth. log_p0: the model's log-probs on x_t, already computed.

  Returns None or dict(seq_idx [S], cand [S,k], vals [S,k], gain [S,k]).
  """
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
    vals = x0s.gather(1, cand)
  else:
    V = lp0.size(-1)
    cl = lp0.gather(1, cand.unsqueeze(-1).expand(-1, -1, V))         # [S,k,V]
    vals = _sample_rows(cl.float().exp().view(-1, V)).view(S, k)

  ce0 = -lp0.gather(-1, x0s.unsqueeze(-1)).squeeze(-1).float()       # [S,L]

  xc = xs.unsqueeze(1).repeat(1, k, 1)
  xc.scatter_(2, cand.unsqueeze(-1), vals.unsqueeze(-1))
  xc, tc = xc.view(S * k, L), ts.repeat_interleave(k)
  tgt = x0s.repeat_interleave(k, dim=0)
  ce_c = []
  for i in range(0, S * k, chunk):
    lp = logprob_fn(xc[i:i + chunk], tc[i:i + chunk])
    ce_c.append(-lp.gather(-1, tgt[i:i + chunk].unsqueeze(-1))
                .squeeze(-1).float())
  ce_c = torch.cat(ce_c).view(S, k, L)

  rem = ms.unsqueeze(1).repeat(1, k, 1)
  rem.scatter_(2, cand.unsqueeze(-1), False)
  rem = rem.float()
  gain = ((ce0.unsqueeze(1) - ce_c) * rem).sum(-1) / rem.sum(-1).clamp(min=1)
  return {'seq_idx': seq_idx, 'cand': cand, 'vals': vals, 'gain': gain}


@torch.no_grad()
def label_batch(model, capture, x0, eps, k, use_ground_truth, chunk):
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
                          k=k, use_ground_truth=use_ground_truth, chunk=chunk)
  if out is None:
    return None
  si, cand = out['seq_idx'], out['cand']
  gather = lambda a: a[si].gather(
    1, cand.unsqueeze(-1).expand(-1, -1, a.size(-1)))
  feats = gather(token_features(log_p0, t))                          # [S,k,F]
  hid = gather(hidden)                                               # [S,k,H]
  V = log_p0.size(-1)
  lp_c = log_p0[si].gather(1, cand.unsqueeze(-1).expand(-1, -1, V))  # [S,k,V]
  cand_logp = lp_c.gather(-1, out['vals'].unsqueeze(-1)).squeeze(-1) # [S,k]
  return {
    'x0': x0[si].int().cpu(), 'mask': (x_t[si] == model.mask_index).cpu(),
    't': t[si].float().cpu(), 'cand': cand.int().cpu(),
    'gain': out['gain'].float().cpu(), 'feats': feats.float().cpu(),
    'hidden': hid.half().cpu(),
    'cand_prob': cand_logp.float().exp().cpu()}  # prob of the sampled token


def load_shards(files):
  shards = [torch.load(f) for f in files]
  return {key: torch.cat([s[key] for s in shards]) for key in shards[0]}


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


def owt_chunks(tokenizer, seq_len, skip=0):
  """Stream OpenWebText as MDLM-style wrapped chunks: [BOS] + text + [EOS],
  documents joined by EOS (BOS = EOS for GPT-2). Yields lists of token ids."""
  import datasets
  ds = datasets.load_dataset('Skylion007/openwebtext', split='train',
                             streaming=True)
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
