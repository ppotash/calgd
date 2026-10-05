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

  def __init__(self, V, L, d, n_layers, n_heads, bits):
    super().__init__()
    self.V, self.bits = V, bits
    self.tok = nn.Embedding(V + 1, d)                      # id V = MASK
    nn.init.normal_(self.tok.weight, std=0.02)
    self.pos = nn.Parameter(torch.randn(L, d) * 0.02)
    self.body = _blocks(d, n_layers, n_heads)
    self.norm = nn.LayerNorm(d)
    # created last so every arm's trunk has the same initialisation; small
    # init so z starts at the scale of the (std 0.02) token embeddings
    self.z_proj = nn.Linear(bits, d) if bits else None
    if self.z_proj is not None:
      nn.init.normal_(self.z_proj.weight, std=0.02)
      nn.init.zeros_(self.z_proj.bias)

  def hidden(self, x, z=None):
    h = self.tok(x) + self.pos[:x.shape[1]]
    if self.z_proj is not None:
      h = h + self.z_proj(2 * z - 1).unsqueeze(1)
    return self.norm(self.body(h))

  def logits(self, h):                                     # tied output layer
    return h @ self.tok.weight[:self.V].T


class Encoder(nn.Module):
  """q(z|x0): shares the denoiser's token embedding, small transformer, K bits."""

  def __init__(self, denoiser, L, d_enc, n_layers, n_heads, bits):
    super().__init__()
    self.den = [denoiser]                                  # not a submodule
    d = denoiser.tok.weight.shape[1]
    self.proj = nn.Linear(d, d_enc)
    self.pos = nn.Parameter(torch.randn(L, d_enc) * 0.02)
    self.body = _blocks(d_enc, n_layers, n_heads)
    self.norm = nn.LayerNorm(d_enc)
    self.head = nn.Linear(d_enc, bits)

  def forward(self, x0):
    h = self.proj(self.den[0].tok(x0)) + self.pos[:x0.shape[1]]
    return self.head(self.norm(self.body(h)).mean(1))


def bernoulli_st(logits):
  p = torch.sigmoid(logits)
  return torch.bernoulli(p.detach()) + p - p.detach(), p


def kl_bits(p):
  p = p.clamp(1e-6, 1 - 1e-6)
  return p * (2 * p).log() + (1 - p) * (2 * (1 - p)).log()   # [B, K] nats


def diffusion_nll(model, x0, z, t=None, chunk=8192):
  """MDLM loss, linear schedule: (1/t) * sum of CE over masked tokens. [B]"""
  B, L = x0.shape
  if t is None:   # stratified
    u = torch.rand(1, device=x0.device)
    t = ((u + torch.arange(B, device=x0.device) / B) % 1.0)
  t = t.clamp(min=1e-3)
  masked = torch.rand(B, L, device=x0.device) < t[:, None]
  xt = torch.where(masked, model.V, x0)
  h = model.hidden(xt, z)
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
  model = Denoiser(V, args.seq_len, args.d, args.layers, args.heads, args.bits)
  enc = (Encoder(model, args.seq_len, args.enc_d, args.enc_layers, args.heads,
                 args.bits) if args.bits else None)
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
  for source in (['encoder', 'prior'] if enc is not None else ['none']):
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
          elif source == 'prior':
            z = torch.bernoulli(torch.full((64, args.bits), 0.5, device=device),
                                generator=gz)
          with torch.autocast(device, dtype=torch.bfloat16, enabled=device == 'cuda'):
            nll += diffusion_nll(model, x0, z).sum().item()
    n_tok = 2 * n_seqs * args.seq_len
    if source == 'prior':
      out['nll_prior_z'] = nll / n_tok
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
        + (f"  (prior z {m['nll_prior_z']:.4f}, KL {m['kl_per_seq']:.2f} nats/seq)"
           if enc else ''))

  model.train()
  t0, run = time.time(), [0.0, 0.0, 0]
  while step < args.steps:
    for g in opt.param_groups:
      g['lr'] = lr_at(step)
    x0 = data.batch(args.batch, step, args.seed).to(device, non_blocking=True)
    with torch.autocast(device, dtype=torch.bfloat16, enabled=device == 'cuda'):
      if enc is not None:
        z, p = bernoulli_st(enc(x0).float())
        kl = kl_bits(p).sum(-1)
      else:
        z, kl = None, torch.zeros(args.batch, device=device)
      nll = diffusion_nll(model, x0, z)
    beta = args.beta * min(1.0, (step + 1) / max(1, args.kl_warmup * args.steps))
    loss = (nll.mean() + beta * kl.mean()) / args.seq_len
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
# Sample + metrics
# ----------------------------------------------------------------------------
@torch.no_grad()
def sample(model, n, L, steps, bits, device, batch=128, row_chunk=2048):
  """Ancestral sampling, linear schedule: each masked token is revealed with
  prob (t - s) / t per step; revealed values drawn in float64 from p(x0|xt,z)."""
  out = []
  for i in range(0, n, batch):
    b = min(batch, n - i)
    x = torch.full((b, L), model.V, device=device)
    z = (torch.bernoulli(torch.full((b, bits), 0.5, device=device))
         if bits else None)
    for k in range(steps):
      t, s = 1 - k / steps, 1 - (k + 1) / steps
      reveal = (x == model.V) & (torch.rand(b, L, device=device) < (t - s) / t)
      if not reveal.any():
        continue
      with torch.autocast(device, dtype=torch.bfloat16, enabled=device == 'cuda'):
        h = model.hidden(x, z)[reveal]
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
    nll, n = 0.0, 0
    for i in range(0, len(x), batch):
      xb = x[i:i + batch].to(device)
      inp = torch.cat([torch.full((len(xb), 1), eos, device=device), xb], 1)
      logits = scorer(inp).logits[:, :-1].float()
      nll += F.cross_entropy(logits.transpose(1, 2), xb, reduction='sum').item()
      n += xb.numel()
    out['gen_ppl'] = math.exp(nll / n)
  return out


def load_scorer(device):
  import transformers
  return transformers.AutoModelForCausalLM.from_pretrained('gpt2-large').to(device).eval()


def sample_cmd(args, device, log, scorer=None):
  data = TokenData(args.out, args.seq_len)
  d = arm_dir(args)
  ck = torch.load(os.path.join(d, 'ckpt.pt'), map_location='cpu')
  a = argparse.Namespace(**{**vars(args), **{k: ck['args'][k] for k in
                            ('d', 'layers', 'heads', 'bits', 'enc_d', 'enc_layers', 'seq_len')}})
  model, _ = build(a, data.vocab)
  model.load_state_dict(ck['model'])
  n_model = len(list(model.parameters()))
  with torch.no_grad():   # EMA weights (the denoiser's share of the shadow list)
    for p, s in zip(model.parameters(), ck['ema'][:n_model]):
      p.copy_(s)
  model.to(device).eval()
  scorer = scorer or load_scorer(device)
  torch.manual_seed(args.seed + 7)
  res = {'arm': os.path.basename(d), 'step': ck['step'], 'by_steps': {}}
  ref_path = os.path.join(args.out, 'reference.json')
  if not os.path.exists(ref_path):
    real = data.batch(args.n_samples, 0, args.seed, 'val')
    ref = text_metrics(real, data.eos, scorer, device)
    json.dump(ref, open(ref_path, 'w'), indent=1)
    log(f"real held-out text: gen_ppl {ref['gen_ppl']:.1f}  entropy "
        f"{ref['entropy']:.3f}  rep4 {ref['rep4']:.3f}")
  for steps in args.sample_steps:
    t0 = time.time()
    x = sample(model, args.n_samples, a.seq_len, steps, a.bits, device)
    m = text_metrics(x, data.eos, scorer, device)
    m['sec'] = time.time() - t0
    res['by_steps'][steps] = m
    log(f"[{res['arm']}] steps={steps:>3}  gen_ppl {m['gen_ppl']:8.1f}  "
        f"entropy {m['entropy']:.3f}  rep4 {m['rep4']:.3f}")
  json.dump(res, open(os.path.join(d, 'samples.json'), 'w'), indent=1)
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
    line = (f"{d:<14} step {h['step']:>6}  bound {h['bound']:.4f}  nll {h['nll']:.4f}"
            + (f"  prior-z nll {h['nll_prior_z']:.4f}  KL {h['kl_per_seq']:.2f}"
               if 'nll_prior_z' in h else ''))
    log(line)
    sp = os.path.join(p, 'samples.json')
    if os.path.exists(sp):
      s = json.load(open(sp))['by_steps']
      for k in sorted(s, key=int):
        m = s[k]
        log(f"    steps {int(k):>3}: gen_ppl {m['gen_ppl']:8.1f}  entropy "
            f"{m['entropy']:.3f}  rep4 {m['rep4']:.3f}")


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('cmd', choices=['prepare', 'train', 'sample', 'compare'])
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
  elif args.cmd == 'sample':
    sample_cmd(args, device, log)
  else:
    compare_cmd(args, log)


if __name__ == '__main__':
  main()
