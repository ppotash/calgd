#!/usr/bin/env python3
"""Latent bits trained jointly from scratch on real text (small scale).

Does a sampled discrete latent improve few-step sampling of a masked diffusion
LM on real text when the denoiser is trained with it from the start (as in the
synthetic probe), rather than fine-tuned into a pretrained model?

Subcommands (all outputs under --out, normally on Drive):
  prepare  Stream OpenWebText (shuffled), tokenize with GPT-2, write
           tokens.bin (uint16, documents joined by EOS) + meta.json.
  train    Train one arm: --bits 0 (control) or --bits K (latent). Resumable;
           data order is identical across arms. Keeps an EMA for evaluation.
  sample   Few-step ancestral sampling from the EMA model (z ~ prior for latent
           models); gen-PPL under GPT-2 Large, entropy, repeated 4-grams, plus
           the same metrics for real held-out text.
  compare  Table of all trained arms in --out.

Model: bidirectional transformer (pre-LN, SDPA attention, bf16 autocast), tied
input/output embeddings, MDLM loss with linear schedule. Latent: encoder q(z|x0)
-> K straight-through Bernoulli bits, uniform prior, KL weight --beta (0.1; 1
collapses the latent, see the synthetic probe), z added to every token embedding.
Reported bounds use the full KL.

Example (Colab A100):
  python probes/text_scratch.py prepare --out $D --tokens 500_000_000
  python probes/text_scratch.py train   --out $D --bits 0
  python probes/text_scratch.py train   --out $D --bits 16
  python probes/text_scratch.py sample  --out $D --bits 0
  python probes/text_scratch.py sample  --out $D --bits 16
  python probes/text_scratch.py compare --out $D
"""
import argparse
import json
import math
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def prepare(out, n_tokens, seed, texts=None, tokenizer=None, batch_docs=1000,
            log=print):
  """Tokenize streamed OpenWebText into out/tokens.bin (uint16)."""
  os.makedirs(out, exist_ok=True)
  if tokenizer is None:
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
  if texts is None:
    import datasets
    ds = datasets.load_dataset('Skylion007/openwebtext', split='train',
                               streaming=True)
    texts = (ex['text'] for ex in ds.shuffle(seed=seed, buffer_size=10_000))
  eos = tokenizer.eos_token_id
  assert len(tokenizer) <= 65536, 'uint16 storage needs vocab <= 65536'
  path, n, t0, batch = os.path.join(out, 'tokens.bin'), 0, time.time(), []
  next_log = 50_000_000
  with open(path, 'wb') as f:
    def flush(batch):
      ids = tokenizer(batch, return_attention_mask=False)['input_ids']
      arr = np.concatenate([np.asarray(doc + [eos], dtype=np.uint16)
                            for doc in ids])
      arr.tofile(f)
      return len(arr)
    for text in texts:
      batch.append(text)
      if len(batch) == batch_docs:
        n += flush(batch)
        batch = []
        if n >= next_log or n >= n_tokens:
          log(f'  {n / 1e6:.0f}M tokens ({n / (time.time() - t0) / 1e6:.2f}M/s)')
          next_log += 50_000_000
        if n >= n_tokens:
          break
    if batch and n < n_tokens:
      n += flush(batch)
  json.dump({'n_tokens': n, 'vocab': len(tokenizer), 'eos': eos,
             'seed': seed}, open(os.path.join(out, 'meta.json'), 'w'))
  log(f'Wrote {n:,} tokens to {path}')


class TokenData:
  """Random fixed-length windows from tokens.bin; the last 1% is held out."""

  def __init__(self, out, seq_len, local_copy=True):
    meta = json.load(open(os.path.join(out, 'meta.json')))
    path = os.path.join(out, 'tokens.bin')
    if local_copy and not path.startswith('/tmp') and os.path.isdir('/content'):
      local = '/content/text_scratch_tokens.bin'   # Drive reads are slow
      if not os.path.exists(local) or os.path.getsize(local) != os.path.getsize(path):
        shutil.copyfile(path, local)
      path = local
    self.tokens = np.memmap(path, dtype=np.uint16, mode='r')
    self.vocab, self.eos, self.L = meta['vocab'], meta['eos'], seq_len
    self.split = int(len(self.tokens) * 0.99)

  def batch(self, n, step, seed, split='train'):
    """Deterministic in (seed, step, split): identical for every arm."""
    g = np.random.default_rng([seed, step, 0 if split == 'train' else 1])
    lo, hi = (0, self.split) if split == 'train' else (self.split, len(self.tokens))
    starts = g.integers(lo, hi - self.L, size=n)
    x = np.stack([self.tokens[s:s + self.L] for s in starts]).astype(np.int64)
    return torch.from_numpy(x)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def _blocks(d, n_layers, n_heads):
  layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout=0.0,
                                     activation='gelu', batch_first=True,
                                     norm_first=True)
  return nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)


class Denoiser(nn.Module):

  def __init__(self, V, L, d, n_layers, n_heads, bits, z_mode='add',
               n_prefix=4):
    super().__init__()
    self.V, self.bits = V, bits
    self.z_mode = z_mode if bits else None
    assert self.z_mode in (None, 'add', 'prefix'), z_mode
    self.tok = nn.Embedding(V + 1, d)                      # id V = MASK
    nn.init.normal_(self.tok.weight, std=0.02)
    self.pos = nn.Parameter(torch.randn(L, d) * 0.02)
    self.body = _blocks(d, n_layers, n_heads)
    self.norm = nn.LayerNorm(d)
    # created last so every arm's trunk has the same initialisation; small
    # init so z starts at the scale of the (std 0.02) token embeddings
    self.z_proj, self.n_prefix = None, 0
    if self.z_mode == 'add':        # z added to every token embedding
      self.z_proj = nn.Linear(bits, d)
    elif self.z_mode == 'prefix':   # z as n_prefix extra tokens to attend to
      self.n_prefix = n_prefix
      self.z_proj = nn.Linear(bits, n_prefix * d)
      self.null_prefix = nn.Parameter(torch.randn(n_prefix, d) * 0.02)
      self.prefix_pos = nn.Parameter(torch.randn(n_prefix, d) * 0.02)
    if self.z_proj is not None:
      nn.init.normal_(self.z_proj.weight, std=0.02)
      nn.init.zeros_(self.z_proj.bias)

  def hidden(self, x, z=None, keep=None):
    """keep [B] in {0,1}: 0 replaces z by the 'no latent' input (all zeros
    after the 2z-1 encoding), used for latent dropout and null-z sampling."""
    h = self.tok(x) + self.pos[:x.shape[1]]
    if self.z_mode == 'add':
      zs = 2 * z - 1
      if keep is not None:
        zs = zs * keep[:, None].to(zs.dtype)
      h = h + self.z_proj(zs).unsqueeze(1)
    elif self.z_mode == 'prefix':
      P = self.n_prefix
      pre = self.z_proj(2 * z - 1).view(len(x), P, -1)
      if keep is not None:   # dropped / null: a learned 'no latent' prefix
        k = keep[:, None, None].to(pre.dtype)
        pre = k * pre + (1 - k) * self.null_prefix.to(pre.dtype)
      h = torch.cat([pre + self.prefix_pos.to(pre.dtype), h], 1)
      return self.norm(self.body(h))[:, P:]
    return self.norm(self.body(h))

  def logits(self, h):                                     # tied output layer
    return h @ self.tok.weight[:self.V].T


class Encoder(nn.Module):
  """q(z|x0): shares the denoiser's token embedding, small transformer, K bits."""

  def __init__(self, denoiser, L, d_enc, n_layers, n_heads, bits,
               detach_emb=True):
    super().__init__()
    self.den = [denoiser]                                  # not a submodule
    self.detach_emb = detach_emb
    d = denoiser.tok.weight.shape[1]
    self.proj = nn.Linear(d, d_enc)
    self.pos = nn.Parameter(torch.randn(L, d_enc) * 0.02)
    self.body = _blocks(d_enc, n_layers, n_heads)
    self.norm = nn.LayerNorm(d_enc)
    self.head = nn.Linear(d_enc, bits)

  def forward(self, x0):
    w = self.den[0].tok.weight
    emb = F.embedding(x0, w.detach() if self.detach_emb else w)
    h = self.proj(emb) + self.pos[:x0.shape[1]]
    return self.head(self.norm(self.body(h)).mean(1))


def bernoulli_st(logits):
  p = torch.sigmoid(logits)
  return torch.bernoulli(p.detach()) + p - p.detach(), p


def kl_bits(p):
  p = p.clamp(1e-6, 1 - 1e-6)
  return p * (2 * p).log() + (1 - p) * (2 * (1 - p)).log()   # [B, K] nats


def diffusion_nll(model, x0, z, t=None, chunk=8192, keep=None):
  """MDLM loss, linear schedule: (1/t) * sum of CE over masked tokens. [B]"""
  B, L = x0.shape
  if t is None:   # stratified
    u = torch.rand(1, device=x0.device)
    t = ((u + torch.arange(B, device=x0.device) / B) % 1.0)
  t = t.clamp(min=1e-3)
  masked = torch.rand(B, L, device=x0.device) < t[:, None]
  xt = torch.where(masked, model.V, x0)
  h = model.hidden(xt, z, keep)
  hm, tgt = h[masked], x0[masked]
  seq = masked.nonzero()[:, 0]
  ce = torch.cat([F.cross_entropy(model.logits(hm[i:i + chunk]).float(),
                                  tgt[i:i + chunk], reduction='none')
                  for i in range(0, len(tgt), chunk)]) if len(tgt) else hm.sum(-1)
  per_seq = torch.zeros(B, device=x0.device).index_add_(0, seq, ce)
  return per_seq / t


class EMA:
  def __init__(self, params, decay):
    self.decay, self.params = decay, list(params)
    self.shadow = [p.detach().clone().float() for p in self.params]

  @torch.no_grad()
  def update(self):
    for s, p in zip(self.shadow, self.params):
      s.lerp_(p.detach().float(), 1 - self.decay)

  @torch.no_grad()
  def swap(self):   # call twice to restore
    for s, p in zip(self.shadow, self.params):
      tmp = p.detach().clone()
      p.copy_(s.to(p.dtype))
      s.copy_(tmp.float())


def build(args, V):
  torch.manual_seed(args.seed)
  model = Denoiser(V, args.seq_len, args.d, args.layers, args.heads, args.bits,
                   getattr(args, 'z_mode', 'add'), getattr(args, 'n_prefix', 4))
  enc = (Encoder(model, args.seq_len, args.enc_d, args.enc_layers, args.heads,
                 args.bits, detach_emb=getattr(args, 'enc_detach', True))
         if args.bits else None)
  return model, enc


def arm_dir(args):
  return os.path.join(args.out, f'bits{args.bits}' + (f'_{args.tag}' if args.tag else ''))


# ----------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, enc, data, args, device, n_seqs=256):
  """Held-out bound per token (full KL) with fixed randomness, so evaluations
  are paired across steps and between arms; plus NLL with z from the prior."""
  model.eval()
  out = {}
  for source in (['encoder', 'prior', 'null'] if enc is not None else ['none']):
    devs = [torch.cuda.current_device()] if device == 'cuda' else []
    with torch.random.fork_rng(devices=devs):
      torch.manual_seed(1234)
      gz = torch.Generator(device=device).manual_seed(4321)
      nll = kl = 0.0
      for rep in range(2):
        for i in range(0, n_seqs, 64):
          x0 = data.batch(64, 10_000 + i + rep * 1000, args.seed, 'val').to(device)
          z = None
          if source == 'encoder':
            p = torch.sigmoid(enc(x0).float())
            z = torch.bernoulli(p, generator=gz)
            kl += kl_bits(p).sum().item()
          elif source in ('prior', 'null'):
            z = torch.bernoulli(torch.full((64, args.bits), 0.5, device=device),
                                generator=gz)
          keep = (torch.zeros(64, device=device) if source == 'null' else None)
          with torch.autocast(device, dtype=torch.bfloat16, enabled=device == 'cuda'):
            nll += diffusion_nll(model, x0, z, keep=keep).sum().item()
    n_tok = 2 * n_seqs * args.seq_len
    if source == 'prior':
      out['nll_prior_z'] = nll / n_tok
    elif source == 'null':
      out['nll_null_z'] = nll / n_tok
    else:
      out['nll'] = nll / n_tok
      out['kl_per_seq'] = kl / (2 * n_seqs)
      out['bound'] = (nll + kl) / n_tok
  model.train()
  return out


def train(args, device, log):
  data = TokenData(args.out, args.seq_len)
  model, enc = build(args, data.vocab)
  model.to(device)
  params = list(model.parameters()) + (list(enc.to(device).parameters()) if enc else [])
  n_params = sum(p.numel() for p in params)
  decay = [p for p in params if p.ndim >= 2]
  no_decay = [p for p in params if p.ndim < 2]
  opt = torch.optim.AdamW([{'params': decay, 'weight_decay': args.wd},
                           {'params': no_decay, 'weight_decay': 0.0}],
                          lr=args.lr, betas=(0.9, 0.98))
  ema = EMA(params, args.ema)
  d = arm_dir(args)
  os.makedirs(d, exist_ok=True)
  ck_path, hist_path = os.path.join(d, 'ckpt.pt'), os.path.join(d, 'history.json')
  step, history = 0, []
  if os.path.exists(ck_path):
    ck = torch.load(ck_path, map_location='cpu')
    model.load_state_dict(ck['model'])
    if enc:
      enc.load_state_dict(ck['enc'])
    opt.load_state_dict(ck['opt'])
    ema.shadow = [s.to(device) for s in ck['ema']]
    step = ck['step']
    history = json.load(open(hist_path)) if os.path.exists(hist_path) else []
    log(f'Resumed {d} at step {step}')
  log(f'arm {os.path.basename(d)}: {n_params / 1e6:.1f}M params, '
      f'{args.steps} steps x {args.batch} x {args.seq_len} tokens '
      f'= {args.steps * args.batch * args.seq_len / 1e6:.0f}M tokens')

  def lr_at(s):
    if s < args.warmup:
      return args.lr * (s + 1) / args.warmup
    frac = (s - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))

  def save():
    torch.save({'model': model.state_dict(),
                'enc': enc.state_dict() if enc else None,
                'opt': opt.state_dict(), 'ema': [s.cpu() for s in ema.shadow],
                'step': step, 'args': vars(args)}, ck_path + '.tmp')
    os.replace(ck_path + '.tmp', ck_path)
    json.dump(history, open(hist_path, 'w'), indent=1)

  def do_eval():
    ema.swap()
    m = evaluate(model, enc, data, args, device)
    ema.swap()
    m['step'] = step
    history.append(m)
    log(f"[eval {step}] bound/tok {m['bound']:.4f}  nll/tok {m['nll']:.4f}"
        + (f"  (prior z {m['nll_prior_z']:.4f}, no z {m['nll_null_z']:.4f}, "
           f"KL {m['kl_per_seq']:.2f} nats/seq)" if enc else ''))

  model.train()
  t0, run = time.time(), [0.0, 0.0, 0]
  while step < args.steps:
    for g in opt.param_groups:
      g['lr'] = lr_at(step)
    x0 = data.batch(args.batch, step, args.seed).to(device, non_blocking=True)
    with torch.autocast(device, dtype=torch.bfloat16, enabled=device == 'cuda'):
      keep, kl_term = None, torch.zeros((), device=device)
      if enc is not None:
        z, p = bernoulli_st(enc(x0).float())
        kb = kl_bits(p)                                     # [B, K]
        keep = torch.ones(args.batch, device=device)
        if args.z_dropout > 0:   # the denoiser must also work without z
          keep = (torch.rand(args.batch, device=device) >= args.z_dropout).float()
        if step < args.z_start:  # latent switched off early in training
          keep = torch.zeros_like(keep)
        kl = (kb.sum(-1) * keep)                            # logged; 0 if dropped
        n_keep = keep.sum().clamp(min=1)
        per_bit = (kb * keep[:, None]).sum(0) / n_keep      # mean over kept seqs
        if args.free_bits > 0:   # no KL pressure until a bit carries free_bits
          per_bit = per_bit.clamp(min=args.free_bits)
        kl_term = per_bit.sum() * (keep.sum() > 0) * (n_keep / args.batch)
      else:
        z, kl = None, torch.zeros(args.batch, device=device)
      nll = diffusion_nll(model, x0, z, keep=keep)
    beta = args.beta * min(1.0, (step + 1) / max(1, args.kl_warmup * args.steps))
    loss = (nll.mean() + beta * kl_term) / args.seq_len
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    ema.update()
    step += 1
    run[0] += nll.mean().item() / args.seq_len
    run[1] += kl.mean().item()
    run[2] += 1
    if step % args.log_every == 0:
      dt = time.time() - t0
      log(f'step {step:>6}/{args.steps}  nll/tok {run[0] / run[2]:.4f}  '
          f'kl {run[1] / run[2]:.2f}  lr {lr_at(step):.2e}  '
          f'{run[2] * args.batch * args.seq_len / dt / 1e3:.0f}k tok/s')
      t0, run = time.time(), [0.0, 0.0, 0]
    if step % args.eval_every == 0 or step == args.steps:
      do_eval()
      save()
  if not history or history[-1]['step'] != step:
    do_eval()
    save()
  log(f'Done: {ck_path}')


# ----------------------------------------------------------------------------
# Learned prior over codes
# ----------------------------------------------------------------------------
class ARBitsPrior(nn.Module):
  """p(z) = prod_k p(z_k | z_<k), each a logistic regression on earlier bits."""

  def __init__(self, bits):
    super().__init__()
    self.bits = bits
    self.w = nn.Parameter(torch.zeros(bits, bits))
    self.b = nn.Parameter(torch.zeros(bits))
    self.register_buffer('mask', torch.tril(torch.ones(bits, bits), -1))

  def logits(self, z):
    return (2 * z - 1) @ (self.w * self.mask).T + self.b

  def log_prob(self, z):
    return -F.binary_cross_entropy_with_logits(
      self.logits(z), z, reduction='none').sum(-1)

  @torch.no_grad()
  def sample(self, n, device):
    z = torch.zeros(n, self.bits, device=device)
    for k in range(self.bits):
      z[:, k] = torch.bernoulli(torch.sigmoid(self.logits(z)[:, k]))
    return z


class CategoricalPrior(nn.Module):
  """Smoothed histogram over all 2^K codes."""

  def __init__(self, bits, counts=None, alpha=1.0):
    super().__init__()
    self.bits = bits
    c = counts if counts is not None else torch.zeros(2 ** bits)
    self.register_buffer('logp', ((c + alpha) / (c + alpha).sum()).log())
    self.register_buffer('pow2', 2 ** torch.arange(bits))

  def log_prob(self, z):
    return self.logp[(z * self.pow2).sum(-1).long()]

  @torch.no_grad()
  def sample(self, n, device):
    codes = torch.multinomial(self.logp.exp(), n, replacement=True)
    return ((codes[:, None] // self.pow2) % 2).float().to(device)


def load_arm(args, device):
  """EMA denoiser + encoder of an arm, and its training args."""
  data = TokenData(args.out, args.seq_len)
  d = arm_dir(args)
  ck = torch.load(os.path.join(d, 'ckpt.pt'), map_location='cpu')
  a = argparse.Namespace(**{**vars(args), **{k: ck['args'][k] for k in
                            ('d', 'layers', 'heads', 'bits', 'enc_d', 'enc_layers',
                             'seq_len', 'z_mode', 'n_prefix') if k in ck['args']}})
  model, enc = build(a, data.vocab)
  model.load_state_dict(ck['model'])
  n_model = len(list(model.parameters()))
  with torch.no_grad():   # EMA weights: denoiser first, then the encoder
    for p, s in zip(model.parameters(), ck['ema'][:n_model]):
      p.copy_(s)
    if enc is not None:
      enc.load_state_dict(ck['enc'])
      for p, s in zip(enc.parameters(), ck['ema'][n_model:]):
        p.copy_(s)
      enc.to(device).eval()
  model.to(device).eval()
  return data, d, ck, a, model, enc


@torch.no_grad()
def encode_windows(enc, data, n, step0, seed, split, device):
  """q(z|x) probabilities for n windows."""
  ps = []
  for i in range(0, n, 256):
    x = data.batch(min(256, n - i), step0 + i, seed, split).to(device)
    ps.append(torch.sigmoid(enc(x).float()))
  return torch.cat(ps)


def fit_prior_cmd(args, device, log):
  """Fit priors over codes to the encoder's posterior samples on training
  text; keep the one with the best held-out log-likelihood. Also estimates
  KL(q || prior) and the bound it implies."""
  data, d, ck, a, model, enc = load_arm(args, device)
  assert enc is not None, 'fit_prior needs a latent arm'
  K = a.bits
  p_tr = encode_windows(enc, data, args.prior_train, 300_000, args.seed, 'train', device)
  p_va = encode_windows(enc, data, args.prior_val, 400_000, args.seed, 'val', device)
  g = torch.Generator(device=device).manual_seed(0)
  z_tr = torch.bernoulli(p_tr, generator=g)
  z_va = torch.bernoulli(p_va, generator=g)
  logq_va = (z_va * p_va.clamp_min(1e-9).log()
             + (1 - z_va) * (1 - p_va).clamp_min(1e-9).log()).sum(-1)
  codes = (z_tr * (2 ** torch.arange(K, device=device))).sum(1).long()
  log(f'{len(z_tr)} training codes: {len(torch.unique(codes))} distinct of {2 ** K}; '
      f'mean bit certainty {(2 * (p_tr - 0.5).abs()).mean():.2f}')

  cands = {'uniform': CategoricalPrior(K, alpha=1.0).to(device)}
  rates = z_tr.mean(0).clamp(1e-4, 1 - 1e-4)
  indep = ARBitsPrior(K).to(device)
  with torch.no_grad():
    indep.b.copy_((rates / (1 - rates)).log())
  cands['independent bits'] = indep
  ar = ARBitsPrior(K).to(device)
  opt = torch.optim.Adam(ar.parameters(), lr=0.05)
  for it in range(500):
    idx = torch.randint(0, len(z_tr), (4096,), device=device)
    loss = -ar.log_prob(z_tr[idx]).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
  cands['autoregressive bits'] = ar
  counts = torch.bincount(codes, minlength=2 ** K).float()
  for alpha in (0.01, 0.1, 1.0):
    cands[f'histogram (alpha={alpha})'] = CategoricalPrior(K, counts, alpha).to(device)

  hist = json.load(open(os.path.join(d, 'history.json')))[-1]
  best, best_nll, rows = None, float('inf'), []
  for name, prior in cands.items():
    with torch.no_grad():
      lp = prior.log_prob(z_va)
    nll = -lp.mean().item()
    kl = (logq_va - lp).mean().item()
    rows.append((name, nll, kl))
    if nll < best_nll:
      best, best_nll = name, nll
  log(f"{'prior':<24}{'held-out NLL of codes':>24}{'KL(q||prior)':>15}"
      f"{'bound/tok':>12}   (nats; uniform = {K * math.log(2):.2f})")
  for name, nll, kl in rows:
    log(f"{name:<24}{nll:>24.2f}{kl:>15.2f}{hist['nll'] + kl / a.seq_len:>12.4f}"
        + ('   <- best' if name == best else ''))
  log(f"(bound/tok = encoder-z NLL {hist['nll']:.4f} + KL/{a.seq_len}; "
      f"same model with z off: {hist.get('nll_null_z', float('nan')):.4f})")
  prior = cands[best]
  torch.save({'type': type(prior).__name__, 'name': best, 'bits': K,
              'state': prior.state_dict()}, os.path.join(d, 'prior.pt'))
  log(f"Saved {best!r} prior to {os.path.join(d, 'prior.pt')}")


def load_prior(d, device):
  ck = torch.load(os.path.join(d, 'prior.pt'), map_location='cpu')
  prior = (ARBitsPrior(ck['bits']) if ck['type'] == 'ARBitsPrior'
           else CategoricalPrior(ck['bits']))
  prior.load_state_dict(ck['state'])
  return prior.to(device).eval(), ck['name']


# ----------------------------------------------------------------------------
# Sample + metrics
# ----------------------------------------------------------------------------
@torch.no_grad()
def sample(model, n, L, steps, bits, device, batch=128, row_chunk=2048,
           null_z=False, z_pool=None, prior=None):
  """Ancestral sampling, linear schedule: each masked token is revealed with
  prob (t - s) / t per step; revealed values drawn in float64 from p(x0|xt,z)."""
  out = []
  for i in range(0, n, batch):
    b = min(batch, n - i)
    x = torch.full((b, L), model.V, device=device)
    if bits and z_pool is not None:   # codes of real documents
      z = z_pool[torch.randint(0, len(z_pool), (b,), device=z_pool.device)].to(device)
    elif bits and prior is not None:  # learned prior over codes
      z = prior.sample(b, device)
    else:
      z = (torch.bernoulli(torch.full((b, bits), 0.5, device=device))
           if bits else None)
    keep = torch.zeros(b, device=device) if (bits and null_z) else None
    for k in range(steps):
      t, s = 1 - k / steps, 1 - (k + 1) / steps
      reveal = (x == model.V) & (torch.rand(b, L, device=device) < (t - s) / t)
      if not reveal.any():
        continue
      with torch.autocast(device, dtype=torch.bfloat16, enabled=device == 'cuda'):
        h = model.hidden(x, z, keep)[reveal]
      vals = []
      for j in range(0, len(h), row_chunk):
        probs = F.softmax(model.logits(h[j:j + row_chunk]).double(), -1)
        vals.append(torch.multinomial(probs, 1).squeeze(-1))
      x[reveal] = torch.cat(vals)
    out.append(x)
  return torch.cat(out)


@torch.no_grad()
def text_metrics(x, eos, scorer=None, device='cuda', batch=16):
  """gen-PPL under the scorer (GPT-2 Large; EOS as context for the first
  token), unigram entropy per sample, fraction of repeated 4-grams."""
  ents, reps = [], []
  for row in x.cpu():
    _, c = torch.unique(row, return_counts=True)
    p = c.double() / c.sum()
    ents.append(float(-(p * p.log()).sum()))
    grams = [tuple(row[i:i + 4].tolist()) for i in range(len(row) - 3)]
    reps.append(1 - len(set(grams)) / len(grams))
  out = {'entropy': float(np.mean(ents)), 'rep4': float(np.mean(reps))}
  if scorer is not None:
    per = []
    for i in range(0, len(x), batch):
      xb = x[i:i + batch].to(device)
      inp = torch.cat([torch.full((len(xb), 1), eos, device=device), xb], 1)
      logits = scorer(inp).logits[:, :-1].float()
      per += F.cross_entropy(logits.transpose(1, 2), xb,
                             reduction='none').mean(1).tolist()
    out['gen_ppl'] = math.exp(float(np.mean(per)))
    out['sample_nll'] = per                 # per-sample mean NLL, for bootstraps
  return out


def load_scorer(device):
  import transformers
  return transformers.AutoModelForCausalLM.from_pretrained('gpt2-large').to(device).eval()


def sample_cmd(args, device, log, scorer=None):
  data, d, ck, a, model, enc = load_arm(args, device)
  z_pool, prior = None, None
  if args.z_from == 'data' and a.bits:
    xs_p = encode_windows(enc, data, args.z_pool, 50_000, args.seed, 'val', device)
    z_pool = torch.bernoulli(xs_p)
    codes = (z_pool * (2 ** torch.arange(a.bits, device=device))).sum(1)
    log(f'z from {len(z_pool)} real held-out windows: {len(torch.unique(codes))} '
        f'distinct codes; bit rates {z_pool.mean(0).min():.2f}-{z_pool.mean(0).max():.2f}; '
        f'mean bit certainty {(2 * (xs_p - 0.5).abs()).mean():.2f}')
  elif args.z_from == 'learned' and a.bits:
    prior, pname = load_prior(d, device)
    log(f'z from the learned prior ({pname})')
  scorer = scorer or load_scorer(device)
  torch.manual_seed(args.seed + 7)
  suffix = ('_nullz' if args.null_z else '_dataz' if z_pool is not None
            else '_learnedz' if prior is not None else '')
  res = {'arm': os.path.basename(d) + suffix, 'step': ck['step'], 'by_steps': {}}
  ref_path = os.path.join(args.out, 'reference.json')
  if not os.path.exists(ref_path):
    real = data.batch(args.n_samples, 0, args.seed, 'val')
    ref = text_metrics(real, data.eos, scorer, device)
    json.dump(ref, open(ref_path, 'w'), indent=1)
    log(f"real held-out text: gen_ppl {ref['gen_ppl']:.1f}  entropy "
        f"{ref['entropy']:.3f}  rep4 {ref['rep4']:.3f}")
  for steps in args.sample_steps:
    t0 = time.time()
    x = sample(model, args.n_samples, a.seq_len, steps, a.bits, device,
               null_z=args.null_z, z_pool=z_pool, prior=prior)
    m = text_metrics(x, data.eos, scorer, device)
    m['sec'] = time.time() - t0
    res['by_steps'][steps] = m
    log(f"[{res['arm']}] steps={steps:>3}  gen_ppl {m['gen_ppl']:8.1f}  "
        f"entropy {m['entropy']:.3f}  rep4 {m['rep4']:.3f}")
  json.dump(res, open(os.path.join(d, f'samples{suffix}.json'), 'w'), indent=1)
  if args.show:
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained('gpt2')
    log('example (fewest steps): ' + repr(tok.decode(x[0].tolist())[:400]))


def compare_cmd(args, log):
  ref_path = os.path.join(args.out, 'reference.json')
  if os.path.exists(ref_path):
    r = json.load(open(ref_path))
    log(f"real text: gen_ppl {r['gen_ppl']:.1f}  entropy {r['entropy']:.3f}  rep4 {r['rep4']:.3f}")
  for d in sorted(os.listdir(args.out)):
    p = os.path.join(args.out, d)
    if not os.path.isdir(p) or not d.startswith('bits'):
      continue
    h = json.load(open(os.path.join(p, 'history.json')))[-1]
    line = (f"{d:<16} step {h['step']:>6}  bound {h['bound']:.4f}  nll {h['nll']:.4f}"
            + (f"  prior-z nll {h['nll_prior_z']:.4f}" if 'nll_prior_z' in h else '')
            + (f"  no-z nll {h['nll_null_z']:.4f}" if 'nll_null_z' in h else '')
            + (f"  KL {h['kl_per_seq']:.2f}" if 'kl_per_seq' in h else ''))
    log(line)
    for name, label in (('samples.json', 'z ~ prior' if 'nll_prior_z' in h else ''),
                        ('samples_learnedz.json', 'z ~ learned prior'),
                        ('samples_dataz.json', 'z from real docs'),
                        ('samples_nullz.json', 'no z')):
      sp = os.path.join(p, name)
      if os.path.exists(sp):
        s = json.load(open(sp))['by_steps']
        for k in sorted(s, key=int):
          m = s[k]
          log(f"    steps {int(k):>3}: gen_ppl {m['gen_ppl']:8.1f}  entropy "
              f"{m['entropy']:.3f}  rep4 {m['rep4']:.3f}  {label}")


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('cmd', choices=['prepare', 'train', 'fit_prior', 'sample',
                                  'compare'])
  ap.add_argument('--out', required=True)
  ap.add_argument('--tokens', type=int, default=500_000_000)
  ap.add_argument('--seq_len', type=int, default=256)
  ap.add_argument('--d', type=int, default=512)
  ap.add_argument('--layers', type=int, default=8)
  ap.add_argument('--heads', type=int, default=8)
  ap.add_argument('--bits', type=int, default=0)
  ap.add_argument('--enc_d', type=int, default=256)
  ap.add_argument('--enc_layers', type=int, default=2)
  ap.add_argument('--beta', type=float, default=0.1)
  ap.add_argument('--kl_warmup', type=float, default=0.1)
  ap.add_argument('--z_mode', choices=['add', 'prefix'], default='add',
                  help='add z to every token embedding, or give it as prefix tokens')
  ap.add_argument('--n_prefix', type=int, default=4)
  ap.add_argument('--z_dropout', type=float, default=0.0,
                  help='fraction of training sequences given no z (latent dropout)')
  ap.add_argument('--free_bits', type=float, default=0.0,
                  help='KL floor per bit (nats) in training; max useful ~0.69')
  ap.add_argument('--z_start', type=int, default=0,
                  help='train without z until this step (latent introduced later)')
  ap.add_argument('--enc_detach', type=int, default=1,
                  help='1: encoder reads the token embedding without training it')
  ap.add_argument('--z_from', choices=['prior', 'data', 'learned'], default='prior',
                  help='sample: z uniform, from codes of real held-out text, or '
                       'from the prior fitted by fit_prior')
  ap.add_argument('--prior_train', type=int, default=100_000)
  ap.add_argument('--prior_val', type=int, default=10_000)
  ap.add_argument('--z_pool', type=int, default=1024)
  ap.add_argument('--null_z', action='store_true',
                  help='sample: run a latent model with z switched off')
  ap.add_argument('--steps', type=int, default=15_000)
  ap.add_argument('--batch', type=int, default=128)
  ap.add_argument('--lr', type=float, default=6e-4)
  ap.add_argument('--wd', type=float, default=0.03)
  ap.add_argument('--warmup', type=int, default=1000)
  ap.add_argument('--ema', type=float, default=0.9995)
  ap.add_argument('--log_every', type=int, default=200)
  ap.add_argument('--eval_every', type=int, default=1000)
  ap.add_argument('--sample_steps', nargs='+', type=int, default=[8, 16, 32, 64])
  ap.add_argument('--n_samples', type=int, default=128)
  ap.add_argument('--show', action='store_true')
  ap.add_argument('--tag', default='')
  ap.add_argument('--seed', type=int, default=0)
  args = ap.parse_args()
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  if device == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
  log = lambda s: print(s, flush=True)
  if args.cmd == 'prepare':
    prepare(args.out, args.tokens, args.seed, log=log)
  elif args.cmd == 'train':
    train(args, device, log)
  elif args.cmd == 'fit_prior':
    fit_prior_cmd(args, device, log)
  elif args.cmd == 'sample':
    sample_cmd(args, device, log)
  else:
    compare_cmd(args, log)


if __name__ == '__main__':
  main()
