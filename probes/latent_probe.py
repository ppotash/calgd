#!/usr/bin/env python3
"""Latent-bits probe: does a sampled discrete latent fix few-step parallel decoding?

Synthetic data with a hidden global variable (one of M "modes"), where coherence
is exactly checkable:
  bag:    each mode owns a random subset of tokens; tokens are i.i.d. within a
          sequence given the mode. A sample is valid iff all its tokens belong to
          one mode's subset. Given the mode, positions are independent, so a
          latent that identifies the mode should fully fix 1-step sampling.
  markov: each mode owns a token subset and a sparse Markov chain on it. Valid iff
          the start token and every transition are allowed under one mode. The
          latent can fix mode mixing, but within-mode dependencies still need
          several steps ("local validity" separates the two error types).

Models (small masked diffusion transformers, trained from scratch):
  bits=0: plain masked diffusion (MDLM loss, linear schedule).
  bits=K: + K latent bits. Encoder q(z|x0) (straight-through Bernoulli) in
          training, uniform Bernoulli prior at sampling, KL in the loss (warmed up),
          z added to every token embedding.

Evaluation at 1..64 sampling steps, exact under the true process: validity,
local validity (markov), true NLL/token of samples, mode coverage. Latent models
also report KL and how much of the true mode the encoder's code captures.

Example (Colab, L4): python probes/latent_probe.py --out /content/drive/MyDrive/calgd/probe
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Synthetic data with a hidden mode
# ----------------------------------------------------------------------------
class ModeData:

  def __init__(self, kind, vocab, seq_len, n_modes, subset, n_succ, seed):
    g = torch.Generator().manual_seed(seed)
    self.kind, self.V, self.L, self.M = kind, vocab, seq_len, n_modes
    self.subsets = torch.stack([torch.randperm(vocab, generator=g)[:subset]
                                for _ in range(n_modes)])          # [M, s]
    self.in_subset = torch.zeros(n_modes, vocab, dtype=torch.bool)
    self.in_subset.scatter_(1, self.subsets, True)
    dirichlet = lambda n: torch.distributions.Dirichlet(
      torch.ones(n)).sample()
    torch.manual_seed(seed)
    if kind == 'bag':
      self.emit = torch.zeros(n_modes, vocab)                       # p(tok|z)
      for z in range(n_modes):
        self.emit[z, self.subsets[z]] = dirichlet(subset)
    elif kind == 'markov':
      self.init = torch.zeros(n_modes, vocab)
      self.trans = torch.zeros(n_modes, vocab, vocab)               # p(b|a,z)
      for z in range(n_modes):
        self.init[z, self.subsets[z]] = 1.0 / subset
        for a in self.subsets[z]:
          succ = self.subsets[z][torch.randperm(subset, generator=g)[:n_succ]]
          self.trans[z, a, succ] = dirichlet(n_succ)
      self.allowed_any = (self.trans > 0).any(0)                    # [V, V]
    else:
      raise ValueError(kind)

  def sample(self, n, g=None):
    modes = torch.randint(0, self.M, (n,), generator=g)
    if self.kind == 'bag':
      x = torch.multinomial(self.emit[modes], self.L, replacement=True,
                            generator=g)
    else:
      x = torch.empty(n, self.L, dtype=torch.long)
      x[:, 0] = torch.multinomial(self.init[modes], 1, generator=g).squeeze(1)
      for i in range(1, self.L):
        x[:, i] = torch.multinomial(self.trans[modes, x[:, i - 1]], 1,
                                    generator=g).squeeze(1)
    return x, modes

  @torch.no_grad()
  def evaluate(self, x, eps=1e-3):
    """Exact metrics for samples x [N, L] under the true process."""
    x = x.cpu()
    if self.kind == 'bag':
      p = self.emit[:, x]                                           # [M, N, L]
      valid_z = (p > 0).all(-1)                                     # [M, N]
      logp_z = ((1 - eps) * p + eps / self.V).log().sum(-1)
      local = None
    else:
      a, b = x[:, :-1], x[:, 1:]
      p0 = self.init[:, x[:, 0]]                                    # [M, N]
      pt = self.trans[:, a, b]                                      # [M, N, L-1]
      valid_z = (p0 > 0) & (pt > 0).all(-1)
      logp_z = (((1 - eps) * p0 + eps / self.V).log()
                + ((1 - eps) * pt + eps / self.V).log().sum(-1))
      local = self.allowed_any[a, b].float().mean().item()
    logp = torch.logsumexp(logp_z - math.log(self.M), dim=0)        # [N]
    valid = valid_z.any(0)
    modes = valid_z.float().argmax(0)[valid]
    hist = torch.bincount(modes, minlength=self.M).float()
    if hist.sum() > 0:
      q = hist / hist.sum()
      cov_ent = float(-(q[q > 0] * q[q > 0].log()).sum() / math.log(self.M))
    else:
      cov_ent = 0.0
    return {'valid': valid.float().mean().item(), 'local_valid': local,
            'nll_per_tok': float(-logp.mean() / self.L),
            'modes_covered': int((hist > 0).sum()),
            'coverage_entropy': cov_ent}


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def _transformer(d, n_layers, n_heads):
  layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout=0.0,
                                     activation='gelu', batch_first=True,
                                     norm_first=True)
  return nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)


class Denoiser(nn.Module):
  """Bidirectional transformer predicting x0 logits; optional latent bits."""

  def __init__(self, V, L, d, n_layers, n_heads, bits):
    super().__init__()
    self.V, self.bits = V, bits
    self.tok = nn.Embedding(V + 1, d)                    # last id = MASK
    self.pos = nn.Parameter(torch.randn(L, d) * 0.02)
    self.z_proj = nn.Linear(bits, d) if bits else None
    self.body = _transformer(d, n_layers, n_heads)
    self.norm = nn.LayerNorm(d)
    self.out = nn.Linear(d, V)

  def forward(self, x, z=None):
    h = self.tok(x) + self.pos
    if self.z_proj is not None:
      h = h + self.z_proj(2 * z - 1).unsqueeze(1)
    return self.out(self.norm(self.body(h)))


class Encoder(nn.Module):
  """q(z|x0): K independent Bernoulli bits from the clean sequence."""

  def __init__(self, V, L, d, n_layers, n_heads, bits):
    super().__init__()
    self.tok = nn.Embedding(V, d)
    self.pos = nn.Parameter(torch.randn(L, d) * 0.02)
    self.body = _transformer(d, n_layers, n_heads)
    self.head = nn.Linear(d, bits)

  def forward(self, x0):
    h = self.body(self.tok(x0) + self.pos)
    return self.head(h.mean(1))                          # logits [B, K]


def bernoulli_st(logits):
  """Straight-through Bernoulli sample (hard forward, sigmoid gradient)."""
  p = torch.sigmoid(logits)
  hard = torch.bernoulli(p.detach())
  return hard + p - p.detach(), p


def kl_to_uniform(p):
  """Per-bit KL(Bern(p) || Bern(0.5)) in nats. p: [B, K] -> [B, K]."""
  p = p.clamp(1e-6, 1 - 1e-6)
  return p * (2 * p).log() + (1 - p) * (2 * (1 - p)).log()


def kl_penalty(kl_bits, beta, free_bits):
  """Training KL term: beta * sum_k max(free_bits, batch-mean KL of bit k).
  With beta=1, free_bits=0 this is the exact ELBO term. Below that, the latent
  is pushed to carry information: with a perfect denoiser the ELBO is already
  tight without a latent, so at beta=1 the latent's KL cost exactly cancels
  its benefit and it tends to collapse."""
  per_bit = kl_bits.mean(0)
  if free_bits > 0:
    per_bit = per_bit.clamp(min=free_bits)
  return beta * per_bit.sum()


def diffusion_nll(model, x0, z, mask_id, generator=None):
  """MDLM ELBO term with linear schedule alpha_t = 1 - t:
  E_t[(1/t) * sum over masked positions of CE]. Stratified t per batch."""
  B, L = x0.shape
  u = torch.rand(1, device=x0.device, generator=generator)
  t = ((u + torch.arange(B, device=x0.device) / B) % 1.0).clamp(min=1e-3)
  masked = torch.rand(B, L, device=x0.device, generator=generator) < t[:, None]
  xt = torch.where(masked, mask_id, x0)
  logits = model(xt, z)
  ce = F.cross_entropy(logits.float().transpose(1, 2), x0, reduction='none')
  return (ce * masked).sum(1) / t                                   # [B]


@torch.no_grad()
def sample(model, n, L, steps, bits, device, mask_id, batch=500):
  """Ancestral masked-diffusion sampling (linear schedule): at each step every
  masked token is revealed with prob (t - s) / t; values ~ p(x0 | x_t, z)."""
  out = []
  for i in range(0, n, batch):
    b = min(batch, n - i)
    x = torch.full((b, L), mask_id, device=device)
    z = torch.bernoulli(torch.full((b, bits), 0.5, device=device)) if bits else None
    for k in range(steps):
      t, s = 1 - k / steps, 1 - (k + 1) / steps
      probs = F.softmax(model(x, z).float(), -1)
      cand = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(b, L)
      reveal = (x == mask_id) & (torch.rand(b, L, device=device) < (t - s) / t)
      x = torch.where(reveal, cand, x)
    out.append(x)
  return torch.cat(out)


@torch.no_grad()
def code_mode_info(encoder, x0, modes, M):
  """Fraction of the true mode's entropy captured by the encoder's hard code:
  I(code; mode) / H(mode), from a contingency table."""
  probs = torch.sigmoid(encoder(x0)).cpu()
  bits = (probs > 0.5).long()
  code = (bits * (2 ** torch.arange(bits.shape[1]))).sum(1)
  joint = torch.zeros(int(code.max()) + 1, M)
  joint.index_put_((code, modes.cpu()), torch.ones(len(code)), accumulate=True)
  joint /= joint.sum()
  pc, pm = joint.sum(1, keepdim=True), joint.sum(0, keepdim=True)
  nz = joint > 0
  mi = (joint[nz] * (joint[nz] / (pc @ pm)[nz]).log()).sum()
  hm = -(pm[pm > 0] * pm[pm > 0].log()).sum()
  return float(mi / hm), int((joint.sum(1) > 0).sum())


# ----------------------------------------------------------------------------
# Train + evaluate one configuration
# ----------------------------------------------------------------------------
def run(data, bits, args, device, log):
  torch.manual_seed(args.seed)
  V, L, mask_id = data.V, data.L, data.V
  model = Denoiser(V, L, args.d, args.layers, args.heads, bits).to(device)
  enc = (Encoder(V, L, args.d, args.enc_layers, args.heads, bits).to(device)
         if bits else None)
  params = list(model.parameters()) + (list(enc.parameters()) if enc else [])
  opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
  sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / args.warmup))
  g = torch.Generator().manual_seed(args.seed + 1)
  amp = (torch.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda'
         else torch.autocast('cpu', enabled=False))
  t0, run_nll, run_kl = time.time(), 0.0, 0.0
  for step in range(1, args.steps + 1):
    x0, _ = data.sample(args.batch, g)
    x0 = x0.to(device)
    with amp:
      if enc is not None:
        z, p = bernoulli_st(enc(x0).float())
        kl_bits = kl_to_uniform(p)
        kl = kl_bits.sum(-1)
      else:
        z, kl = None, torch.zeros(args.batch, device=device)
      nll = diffusion_nll(model, x0, z, mask_id)
    if enc is not None:
      ramp = min(1.0, step / max(1, args.kl_warmup * args.steps))
      kl_term = kl_penalty(kl_bits, ramp * args.beta, args.free_bits)
    else:
      kl_term = 0.0
    loss = (nll.mean() + kl_term) / L
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    sched.step()
    run_nll += nll.mean().item() / L
    run_kl += kl.mean().item()
    if step % args.log_every == 0:
      n = args.log_every
      log(f'  [{data.kind} bits={bits}] step {step:>5}  nll/tok {run_nll / n:.3f}'
          f'  kl {run_kl / n:.2f} nats  ({time.time() - t0:.0f}s)')
      run_nll = run_kl = 0.0

  model.eval()
  res = {'data': data.kind, 'bits': bits, 'tag': args.tag,
         'beta': args.beta, 'free_bits': args.free_bits,
         'train_sec': time.time() - t0}
  # held-out ELBO (nats/token, includes KL for latent models)
  ge = torch.Generator().manual_seed(12345)
  xe, me = data.sample(args.eval_n, ge)
  xe = xe.to(device)
  with torch.no_grad():
    nlls, kls = [], []
    for _ in range(8):
      if enc is not None:
        p = torch.sigmoid(enc(xe).float())
        z = torch.bernoulli(p)
        kls.append(kl_to_uniform(p).sum(-1).mean().item())
      else:
        z = None
        kls.append(0.0)
      nlls.append(diffusion_nll(model, xe, z, mask_id).mean().item())
  res['elbo_per_tok'] = (sum(nlls) / 8 + sum(kls) / 8) / L
  res['kl_nats'] = sum(kls) / 8
  if enc is not None:
    enc.eval()
    res['code_mode_info'], res['codes_used'] = code_mode_info(
      enc, xe, me, data.M)
  res['by_steps'] = {}
  for steps in args.sample_steps:
    xs = sample(model, args.n_samples, L, steps, bits, device, mask_id)
    res['by_steps'][steps] = data.evaluate(xs)
  return res


def reference_rows(data, n):
  """True samples, and per-position marginals (the 1-step limit of a perfect
  model without a latent)."""
  g = torch.Generator().manual_seed(999)
  x, _ = data.sample(n, g)
  ref = {'true data': data.evaluate(x)}
  big, _ = data.sample(20000, g)
  marg = torch.stack([torch.bincount(big[:, i], minlength=data.V).float()
                      for i in range(data.L)])
  xm = torch.multinomial(marg / marg.sum(1, keepdim=True), n,
                         replacement=True, generator=g).T
  ref['independent marginals'] = data.evaluate(xm)
  return ref


def _name(r):
  tag = r.get('tag', '')
  return f"bits={r['bits']}" + (f" {tag}" if tag and r['bits'] else '')


def plot(results, refs, args, path):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  kinds = list(dict.fromkeys(r['data'] for r in results))
  metrics = [('valid', 'valid samples (one mode)'),
             ('nll_per_tok', 'true NLL/token of samples'),
             ('coverage_entropy', 'mode coverage (norm. entropy)')]
  fig, axes = plt.subplots(len(kinds), 3, figsize=(15, 4 * len(kinds)),
                           squeeze=False)
  for row, kind in enumerate(kinds):
    for col, (key, title) in enumerate(metrics):
      ax = axes[row, col]
      for r in results:
        if r['data'] != kind:
          continue
        by = {int(k): v for k, v in r['by_steps'].items()}
        st = sorted(by)
        ys = [by[s][key] for s in st]
        ax.plot(st, ys, 'o-', label=_name(r))
      for (name, ref), ls in zip(refs.get(kind, {}).items(), ['--', ':']):
        ax.axhline(ref[key], ls=ls, c='gray', label=name)
      if key in ('valid', 'coverage_entropy'):
        ax.set_ylim(-0.02, 1.02)
      ax.set_xscale('log', base=2)
      ax.set_xlabel('sampling steps')
      ax.set_title(f'{kind}: {title}')
      ax.grid(alpha=0.3)
      if col == 0:
        ax.legend(fontsize=7)
  plt.tight_layout()
  plt.savefig(path, dpi=130)
  plt.close(fig)


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--out', default='probe_out')
  ap.add_argument('--data', nargs='+', default=['bag', 'markov'])
  ap.add_argument('--bits', nargs='+', type=int, default=[0, 2, 4, 8])
  ap.add_argument('--vocab', type=int, default=64)
  ap.add_argument('--seq_len', type=int, default=32)
  ap.add_argument('--modes', type=int, default=16)
  ap.add_argument('--subset', type=int, default=12)
  ap.add_argument('--n_succ', type=int, default=3)
  ap.add_argument('--d', type=int, default=256)
  ap.add_argument('--layers', type=int, default=4)
  ap.add_argument('--enc_layers', type=int, default=2)
  ap.add_argument('--heads', type=int, default=4)
  ap.add_argument('--steps', type=int, default=4000)
  ap.add_argument('--batch', type=int, default=256)
  ap.add_argument('--lr', type=float, default=5e-4)
  ap.add_argument('--warmup', type=int, default=200)
  ap.add_argument('--kl_warmup', type=float, default=0.3,
                  help='fraction of training over which the KL weight ramps up')
  ap.add_argument('--beta', type=float, default=1.0,
                  help='final KL weight (1 = exact ELBO)')
  ap.add_argument('--free_bits', type=float, default=0.0,
                  help='KL floor per bit in nats (max useful ~0.69 = 1 bit)')
  ap.add_argument('--tag', default='',
                  help='label stored with each result (e.g. the KL setting)')
  ap.add_argument('--sample_steps', nargs='+', type=int,
                  default=[1, 2, 4, 8, 16, 32, 64])
  ap.add_argument('--n_samples', type=int, default=1000)
  ap.add_argument('--eval_n', type=int, default=2000)
  ap.add_argument('--log_every', type=int, default=500)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--data_seed', type=int, default=0)
  args = ap.parse_args()

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  os.makedirs(args.out, exist_ok=True)
  res_path = os.path.join(args.out, 'results.json')
  saved = json.load(open(res_path)) if os.path.exists(res_path) else {}
  results = saved.get('results', [])
  done = {(r['data'], r['bits'], r.get('tag', '')) for r in results}
  log = lambda s: print(s, flush=True)
  log(f'device={device}; results -> {res_path}')

  refs = saved.get('refs', {})   # keep references for datasets not run now
  for kind in args.data:
    data = ModeData(kind, args.vocab, args.seq_len, args.modes, args.subset,
                    args.n_succ, args.data_seed)
    refs[kind] = reference_rows(data, args.n_samples)
    for name, r in refs[kind].items():
      log(f'[{kind}] reference {name:<22} valid {r["valid"]:.3f}  '
          f'nll/tok {r["nll_per_tok"]:.3f}'
          + (f'  local {r["local_valid"]:.3f}' if r['local_valid'] is not None
             else ''))
    for bits in args.bits:
      if (kind, bits, args.tag) in done:
        log(f'skip {kind} bits={bits} tag={args.tag!r} (done)')
        continue
      log(f'=== train {kind} bits={bits}')
      r = run(data, bits, args, device, log)
      results.append(r)
      json.dump({'args': vars(args), 'refs': refs, 'results': results},
                open(res_path, 'w'), indent=1)
      extra = (f"  KL {r['kl_nats']:.2f} nats, code captures "
               f"{r['code_mode_info']:.0%} of mode info, {r['codes_used']} codes used"
               if bits else '')
      log(f"  ELBO {r['elbo_per_tok']:.3f} nats/tok{extra}")

  # summary table
  log('\n' + f"{'data':<7}{'model':>16}  " + ''.join(
    f'{s:>8}' for s in args.sample_steps) + '   (fraction of valid samples)')
  for r in results:
    by = {int(k): v for k, v in r['by_steps'].items()}
    log(f"{r['data']:<7}{_name(r):>16}  " + ''.join(
      f"{by[s]['valid']:>8.3f}" if s in by else f"{'-':>8}"
      for s in args.sample_steps))
  if any(r['data'] == 'markov' for r in results):
    log('\nmarkov local validity (transitions allowed under some mode):')
    for r in results:
      if r['data'] == 'markov':
        by = {int(k): v for k, v in r['by_steps'].items()}
        log(f"{'markov':<7}{_name(r):>16}  " + ''.join(
          f"{by[s]['local_valid']:>8.3f}" if s in by else f"{'-':>8}"
          for s in args.sample_steps))
  plot(results, refs, args, os.path.join(args.out, 'probe.png'))
  log(f"\nSaved {res_path} and {os.path.join(args.out, 'probe.png')}")


if __name__ == '__main__':
  main()