import json
import os
import time

import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch

import dataloader
import diffusion
import utils

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(config, tokenizer):
  if 'hf' in config.backbone:
    return diffusion.Diffusion(
      config, tokenizer=tokenizer).to('cuda')
  
  return diffusion.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config)


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)


def generate_samples(config, logger, tokenizer):
  logger.info('Generating samples.')
  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  model.gen_ppl_metric.reset()
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  stride_length = config.sampling.stride_length
  num_strides = config.sampling.num_strides
  for _ in range(config.sampling.num_sample_batches):
    if config.sampling.semi_ar:
      _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
        stride_length=stride_length,
        num_strides=num_strides,
        dt=1 / config.sampling.steps)
      text_samples = intermediate_samples[-1]
      # Note: Samples generated using semi-ar method
      # need to to be processed before computing generative perplexity
      # since these samples contain numerous <|endoftext|> tokens
      # and diffusion.compute_generative_perplexity() discards
      # any text after the first EOS token.
    else:
      samples = model.restore_model_and_sample(
        num_steps=config.sampling.steps)
      text_samples = model.tokenizer.batch_decode(samples)
      model.compute_generative_perplexity(text_samples)
  print('Text samples:', text_samples)
  if not config.sampling.semi_ar:
    print('Generative perplexity:',
          model.gen_ppl_metric.compute())
  return text_samples

def sample_sweep(config, logger, tokenizer):
  """Generate samples at several step counts; record gen-PPL, entropy, time.

  Writes sample_sweep.json to the Hydra run dir after every setting,
  so a disconnect loses at most one setting.
  """
  model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
  if config.eval.disable_ema:
    model.ema = None
  adapter = None
  if config.latent.get('ckpt_path'):
    import latent
    adapter = latent.load_into(model, config.latent.ckpt_path,
                               config.model.length)
    logger.info(f'Loaded latent adapter (bits={adapter.bits}) from '
                f'{config.latent.ckpt_path}')
  z_pool = None
  if adapter is not None and adapter.bits and \
      config.latent.get('sample_z', 'prior') == 'data':
    import revealer
    stream = revealer.owt_chunks(model.tokenizer, config.model.length)
    pool_x = torch.tensor([next(stream)
                           for _ in range(config.latent.z_pool_seqs)])
    z_pool = latent.data_z_pool(adapter, pool_x, model.device)
    logger.info(f'Sampling z from {len(z_pool)} real held-out sequences '
                f'(mean bit rate {z_pool.mean():.2f}) instead of the prior')
  out_path = os.path.join(os.getcwd(), 'sample_sweep.json')
  results = {
    'checkpoint': config.eval.checkpoint_path,
    'predictor': config.sampling.predictor,
    'order': config.sampling.get('order'),
    'fp64': config.sampling.get('fp64', False),
    'seq_len': config.model.length,
    'batch_size': config.loader.eval_batch_size,
    'num_sample_batches': config.sampling.num_sample_batches,
    'seed': config.seed,
    'latent_ckpt': config.latent.get('ckpt_path'),
    'latent_bits': adapter.bits if adapter is not None else 0,
    'sample_z': config.latent.get('sample_z', 'prior'),
    'settings': []}
  for steps in config.sampling.sweep_steps:
    model.gen_ppl_metric.reset()
    texts, ents, secs = [], [], 0.0
    zs, pres = [], []
    for _ in range(config.sampling.num_sample_batches):
      if adapter is not None and adapter.bits:
        if z_pool is not None:
          idx = torch.randint(0, len(z_pool), (config.loader.eval_batch_size,))
          z = z_pool[idx]
        else:
          z = adapter.sample_prior(config.loader.eval_batch_size, model.device)
        adapter.set_z(z)
      torch.cuda.synchronize()
      t0 = time.time()
      samples = model.restore_model_and_sample(num_steps=steps)
      torch.cuda.synchronize()
      secs += time.time() - t0
      ents += diffusion.sample_entropy(samples)
      if adapter is not None and adapter.oracle == 'presence':
        zs.append(z.cpu())
        pres.append(adapter.encoder.presence(samples.to(model.device)).float().cpu())
      batch_texts = model.tokenizer.batch_decode(samples)
      texts += batch_texts
      model.compute_generative_perplexity(batch_texts)
    ent = torch.tensor(ents)
    rec = {'steps': int(steps), 'n_samples': len(texts),
           'gen_ppl': float(model.gen_ppl_metric.compute()),
           'entropy_mean': float(ent.mean()),
           'entropy_std': float(ent.std()) if len(ents) > 1 else 0.0,
           'sec_per_sample': secs / len(texts),
           'samples': texts}
    if zs:   # do samples contain the oracle words iff their bit is on?
      z_all, p_all = torch.cat(zs), torch.cat(pres)
      rec['oracle_present_when_on'] = float(p_all[z_all == 1].mean())
      rec['oracle_present_when_off'] = float(p_all[z_all == 0].mean())
      rec['oracle_agreement'] = float((p_all == z_all).float().mean())
    results['settings'].append(rec)
    with open(out_path, 'w') as f:
      json.dump(results, f, indent=1)
    logger.info(f"steps={rec['steps']:4d}  gen_ppl={rec['gen_ppl']:8.2f}  "
                f"entropy={rec['entropy_mean']:.3f}  "
                f"sec/sample={rec['sec_per_sample']:.2f}"
                + (f"  oracle word present: bit on {rec['oracle_present_when_on']:.2f}"
                   f" / off {rec['oracle_present_when_off']:.2f}"
                   f" (agreement {rec['oracle_agreement']:.2f})"
                   if 'oracle_agreement' in rec else ''))
  print(f"\n{'steps':>6} {'gen_ppl':>9} {'entropy':>8} {'sec/sample':>11}")
  for r in results['settings']:
    print(f"{r['steps']:>6} {r['gen_ppl']:>9.2f} {r['entropy_mean']:>8.3f} "
          f"{r['sec_per_sample']:>11.2f}")
  print(f'Saved to {out_path}')
  return results


def revealer_label(config, logger, tokenizer):
  """Label streamed OpenWebText chunks with counterfactual information gains.

  Writes shard_XXXX.pt files plus progress.json to revealer.label_dir;
  rerunning resumes after the last completed shard.
  """
  import revealer
  rc = config.revealer
  assert rc.label_dir, 'set revealer.label_dir'
  if rc.allow_tf32:   # labels only; sampling evaluations keep full fp32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
  os.makedirs(rc.label_dir, exist_ok=True)
  progress_path = os.path.join(rc.label_dir, 'progress.json')
  progress = (json.load(open(progress_path)) if os.path.exists(progress_path)
              else {'shards': 0, 'consumed': 0})
  if progress['shards'] >= rc.n_shards:
    logger.info(f"All {rc.n_shards} shards already labeled.")
    return
  model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
  model.eval()
  capture = revealer.HiddenCapture(model.backbone)
  logger.info(f'Hidden states captured at: {capture.name}')
  stream = revealer.owt_chunks(model.tokenizer, config.model.length,
                               skip=progress['consumed'])
  torch.manual_seed(rc.seed + progress['shards'])
  consumed = progress['consumed']
  for shard in range(progress['shards'], rc.n_shards):
    recs, n_lab, t0 = [], 0, time.time()
    while n_lab < rc.shard_size:
      batch = [next(stream) for _ in range(rc.batch_size)]
      consumed += len(batch)
      x0 = torch.tensor(batch, device=model.device)
      out = revealer.label_batch(
        model, capture, x0, eps=config.training.sampling_eps, k=rc.k,
        use_ground_truth=rc.ground_truth, chunk=rc.chunk,
        n_samples=rc.n_samples)
      if out is not None:
        recs.append(out)
        n_lab += out['gain'].shape[0]
    data = {key: torch.cat([r[key] for r in recs]) for key in recs[0]}
    torch.save(data, os.path.join(rc.label_dir, f'shard_{shard:04d}.pt'))
    with open(progress_path, 'w') as f:
      json.dump({'shards': shard + 1, 'consumed': consumed}, f)
    g = data['gain']
    logger.info(f'shard {shard + 1}/{rc.n_shards}: {n_lab} seqs, '
                f'{n_lab / (time.time() - t0):.2f} seq/s, '
                f'gain mean {g.mean():.2e} (neg {(g < 0).float().mean():.0%})')


def revealer_train(config, logger, tokenizer):
  """Train the reveal head on labeled shards (no denoiser passes needed).

  revealer.target: 'sampled' ranks candidates by the gain of the specific
  sampled value (what the sampler will commit); 'expected' ranks by the gain
  averaged over the stored samples. revealer.base: None, 'cand' or 'maxp' -
  with a base the head learns a correction on top of that confidence score.
  """
  import revealer
  rc = config.revealer
  files = revealer.shard_files(rc.label_dir)
  assert len(files) > rc.val_shards, (
    f'need more than {rc.val_shards} shards in {rc.label_dir}, '
    f'found {len(files)}')
  val_raw = revealer.load_shards(files[:rc.val_shards])
  tr_raw = revealer.load_shards(files[rc.val_shards:])
  if rc.target == 'expected':
    assert rc.base in (None, 'maxp'), "target=expected supports base null/maxp"
  head_path = rc.head_path or os.path.join(rc.label_dir, 'revealer_head.pt')
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  torch.manual_seed(rc.seed)
  S, k, m = tr_raw['gain'].shape
  logger.info(f"train {S} seqs, val {val_raw['gain'].shape[0]} seqs, k={k}, "
              f"samples per candidate m={m}, target={rc.target}, "
              f"base={rc.base}, use_hidden={rc.use_hidden}")
  if m >= 2:
    pair, r, r_full = revealer.label_reliability(
      torch.cat([tr_raw['gain'], val_raw['gain']]))
    logger.info(f'Label reliability: halves agree on {pair:.3f} of pairs; '
                f'within-seq r={r:.3f} (half vs half), '
                f'~{r_full:.3f} for the full {m}-sample average')

  tr = revealer.group_view(tr_raw, rc.target)
  val = revealer.group_view(val_raw, rc.target)
  vfeats = val_raw['feats'][val['seq']]

  def report(name, score):
    pair, top1 = revealer.ranking_metrics(score, val['gain'], rc.margin)
    bins = []
    for lo, hi in [(0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.01)]:
      sel = (val['t'] >= lo) & (val['t'] < hi)
      if sel.any():
        bins.append(revealer.ranking_metrics(
          score[sel], val['gain'][sel], rc.margin)[0])
    logger.info(f'{name:<30} pair acc {pair:.3f}  top-1 {top1:.3f}  '
                f'by t [low/mid/high] ' + ' / '.join(f'{b:.3f}' for b in bins))
    return pair

  logger.info('Validation baselines (pair acc 0.5 and top-1 1/k = random):')
  report('random', torch.rand_like(val['gain']))
  conf_name = ('confidence (sampled tok)' if rc.target == 'sampled'
               else 'mean sampled-tok prob')
  base_accs = {
    conf_name: report(conf_name, val['cand_prob']),
    'max prob': report('max prob', vfeats[..., 1]),
    'neg entropy': report('neg entropy', -vfeats[..., 0])}
  ref_name = conf_name if rc.base in (None, 'cand') else 'max prob'
  ref_score = val['cand_prob'] if ref_name == conf_name else vfeats[..., 1]

  head = revealer.RevealHead(
    tr_raw['hidden'].shape[-1], width=rc.width, dropout=rc.dropout,
    base=rc.base, use_hidden=rc.use_hidden).to(device)
  opt = torch.optim.AdamW(head.parameters(), lr=rc.lr,
                          weight_decay=rc.weight_decay)

  def scores(view, raw, idx=None):
    seq = view['seq'] if idx is None else view['seq'][idx]
    cp = view['cand_prob'] if idx is None else view['cand_prob'][idx]
    feats = raw['feats'][seq].to(device)
    b = revealer.base_score(rc.base, feats, cp.to(device))
    return head(raw['hidden'][seq].to(device), feats, b)

  def evaluate():
    head.eval()
    with torch.no_grad():
      return scores(val, val_raw).cpu()

  vs = evaluate()
  best = report('head epoch   0 (untrained)', vs)
  best_scores, stale = vs, 0
  head.save(head_path, extra={'epoch': 0, 'val_pair_acc': best})
  G = tr['gain'].shape[0]
  for ep in range(1, rc.epochs + 1):
    head.train()
    perm, tot = torch.randperm(G), 0.0
    for i in range(0, G, rc.train_batch):
      idx = perm[i:i + rc.train_batch]
      loss = revealer.pairwise_rank_loss(
        scores(tr, tr_raw, idx), tr['gain'][idx].to(device), rc.margin)
      opt.zero_grad()
      loss.backward()
      opt.step()
      tot += loss.item() * len(idx)
    vs = evaluate()
    pair = report(f'head epoch {ep:>3} (loss {tot / G:.3f})', vs)
    if pair > best:
      best, best_scores, stale = pair, vs, 0
      head.save(head_path, extra={'epoch': ep, 'val_pair_acc': pair,
                                  'label_dir': rc.label_dir,
                                  'target': rc.target})
    else:
      stale += 1
      if stale >= rc.patience:
        logger.info(f'Early stop: no improvement for {rc.patience} epochs.')
        break

  ch, nv = revealer.pair_counts(best_scores, val['gain'], rc.margin)
  cr, _ = revealer.pair_counts(ref_score, val['gain'], rc.margin)
  lo, hi = revealer.bootstrap_diff(ch, cr, nv, val['seq'])
  ref_acc = (cr.sum() / nv.sum()).item()
  logger.info(f'Best head pair acc {best:.3f} vs {ref_name} {ref_acc:.3f}: '
              f'difference {best - ref_acc:+.3f} (95% CI {lo:+.3f} to {hi:+.3f}). '
              f'Head saved to {head_path}')


def latent_finetune(config, logger, tokenizer):
  """Fine-tune latent bits into the pretrained MDLM (see latent.py).

  Trains the latent encoder, the z projection and the denoiser's adaLN
  modulation layers on streamed OpenWebText; the rest of the denoiser is frozen.
  latent.bits=0 is the matched control (same trainable layers, no latent).
  Saves latest.pt (resumable) and final.pt to latent.save_dir, plus
  metrics.json with the held-out bound over training.
  """
  import latent
  import revealer
  lc = config.latent
  assert lc.save_dir, 'set latent.save_dir'
  os.makedirs(lc.save_dir, exist_ok=True)
  if lc.allow_tf32:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
  torch.manual_seed(lc.seed)
  model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
  model.eval()                          # frozen trunk: no dropout
  model.ema = None
  device, L, B = model.device, config.model.length, lc.batch_size
  for p in model.parameters():
    p.requires_grad_(False)
  scope = lc.get('train_scope', 'adaln')
  assert scope in ('adaln', 'all'), scope
  adaln = latent.adaln_parameters(model.backbone)
  adaln_ids = {id(p) for _, p in adaln}
  trunk = ([p for p in model.backbone.parameters() if id(p) not in adaln_ids]
           if scope == 'all' else [])
  for p in [p for _, p in adaln] + trunk:
    p.requires_grad_(True)
  cond_dim = adaln[0][1].shape[1]       # adaLN maps cond_dim -> k * hidden
  adapter = latent.LatentAdapter(
    model.backbone, lc.bits, model.vocab_size, L, cond_dim,
    lc.enc_dim, lc.enc_layers, lc.enc_heads,
    input_inject=lc.get('input_inject', False),
    head_std=lc.get('enc_head_std', 0.02),
    oracle=lc.get('oracle')).to(device)
  if lc.get('freeze_encoder', False) and lc.bits:
    for p in adapter.encoder.parameters():
      p.requires_grad_(False)
  new_params = [p for p in adapter.parameters() if p.requires_grad]
  groups = [{'params': [p for _, p in adaln], 'lr': lc.lr}]
  if trunk:
    groups.append({'params': trunk, 'lr': lc.trunk_lr})
  if new_params:
    groups.append({'params': new_params, 'lr': lc.lr * lc.new_lr_mult})
  for g in groups:
    g['base_lr'] = g['lr']
  opt = torch.optim.AdamW(groups, weight_decay=0.0)
  n_ad = sum(p.numel() for _, p in adaln) + sum(p.numel() for p in trunk)
  n_new = sum(p.numel() for p in new_params)
  logger.info(f'latent bits={lc.bits} beta={lc.beta} free_bits={lc.free_bits} '
              f"input_inject={lc.get('input_inject', False)} "
              f"enc_head_std={lc.get('enc_head_std', 0.02)} "
              f"freeze_encoder={lc.get('freeze_encoder', False)}; "
              f'conditioning hook at {adapter.hook_name}; scope={scope}; '
              f'trainable: denoiser '
              f'{n_ad / 1e6:.2f}M + latent {n_new / 1e6:.2f}M params')

  latest = os.path.join(lc.save_dir, 'latest.pt')
  metrics_path = os.path.join(lc.save_dir, 'metrics.json')
  step, consumed, history = 0, lc.eval_seqs, []
  if os.path.exists(latest):
    ck = torch.load(latest, map_location='cpu')
    if 'backbone_state' in ck:
      model.backbone.load_state_dict(ck['backbone_state'])
    else:
      params = dict(adaln)
      with torch.no_grad():
        for n, v in ck['adaln'].items():
          params[n].copy_(v.to(device))
    adapter.load_state_dict(ck['adapter'], strict=False)
    if 'opt' in ck:
      opt.load_state_dict(ck['opt'])
    else:
      logger.info('No optimizer state in checkpoint (train_scope=all); '
                  'resuming with a fresh optimizer.')
    step, consumed = ck['step'], ck['consumed']
    history = json.load(open(metrics_path)) if os.path.exists(metrics_path) else []
    logger.info(f'Resumed from step {step}')

  val_stream = revealer.owt_chunks(model.tokenizer, L)
  val = torch.tensor([next(val_stream) for _ in range(lc.eval_seqs)])
  if adapter.oracle == 'presence':
    if step == 0:   # on resume the chosen tokens come from latest.pt
      sel = torch.tensor([next(val_stream) for _ in range(lc.oracle_select_seqs)])
      toks, dfs = latent.select_oracle_tokens(sel, model.tokenizer, lc.bits,
                                              model.vocab_size)
      adapter.encoder.tokens.copy_(toks.to(device))
    toks = adapter.encoder.tokens.tolist()
    words = [model.tokenizer.decode([v]) for v in toks]
    present = adapter.encoder.presence(val.to(device)).float().mean(0).tolist()
    logger.info('oracle bits = presence of: ' + ', '.join(
      f'{w.strip()!r} ({p:.2f})' for w, p in zip(words, present))
      + '  (held-out presence rate)')
  stream = revealer.owt_chunks(model.tokenizer, L, skip=consumed)

  def evaluate():
    """Held-out bound per token (full KL), and how much z helps the decoder.
    Every pass replays the same noise levels and masks (seeded stream), so the
    encoder-z and prior-z passes are paired with each other, across steps, and
    with the control run. z draws use separate generators."""
    adapter.eval()
    devs = [torch.cuda.current_device()] if torch.cuda.is_available() else []

    def run_pass(z_source):
      total_nll, total_kl = 0.0, 0.0
      with torch.random.fork_rng(devices=devs):
        torch.manual_seed(1234)
        gz = torch.Generator(device=device).manual_seed(
          4321 if z_source == 'encoder' else 8765)
        for _ in range(lc.eval_repeats):
          for i in range(0, len(val), B):
            x0 = val[i:i + B].to(device)
            if lc.bits and z_source == 'encoder':
              p = torch.sigmoid(adapter.encoder(x0).float())
              adapter.set_z(torch.bernoulli(p, generator=gz))
              total_kl += latent.kl_to_uniform(p).sum().item()
            elif lc.bits:
              adapter.set_z(adapter.sample_prior(len(x0), device, gz))
            total_nll += model._forward_pass_diffusion(x0).sum().item()
            adapter.set_z(None)
      return total_nll, total_kl

    with torch.no_grad():
      nll, kl = run_pass('encoder')
      nll_prior = run_pass('prior')[0] if lc.bits else nll
    n_seq = len(val) * lc.eval_repeats
    out = {'step': step, 'bound_per_tok': (nll + kl) / (n_seq * L),
           'nll_per_tok': nll / (n_seq * L), 'kl_nats_per_seq': kl / n_seq}
    if lc.bits:
      out['nll_per_tok_prior_z'] = nll_prior / (n_seq * L)
    adapter.train()
    return out

  def log_eval(m):
    extra = (f"  (nll with prior z {m['nll_per_tok_prior_z']:.4f}, "
             f"KL {m['kl_nats_per_seq']:.2f} nats/seq)" if lc.bits else '')
    logger.info(f"[eval step {m['step']}] bound/tok {m['bound_per_tok']:.4f}"
                f"  nll/tok {m['nll_per_tok']:.4f}{extra}")

  def save(path, with_opt):
    extra = {'step': step, 'consumed': consumed,
             'config': omegaconf.OmegaConf.to_container(lc)}
    full = scope == 'all'
    if with_opt and not full:   # full-model optimizer state would be ~1.4 GB
      extra['opt'] = opt.state_dict()
    latent.save_checkpoint(path, adapter, model.backbone, extra, full=full)

  if step == 0:
    m = evaluate()
    history.append(m)
    log_eval(m)
  adapter.train()
  t0, run = time.time(), {'nll': 0.0, 'kl': 0.0, 'n': 0}
  while step < lc.steps:
    frac = (step + 1) / max(1, lc.warmup)
    for g in opt.param_groups:
      g['lr'] = g['base_lr'] * min(1.0, frac)
    beta = lc.beta * min(1.0, (step + 1) / max(1, lc.kl_warmup * lc.steps))
    opt.zero_grad(set_to_none=True)
    for _ in range(lc.grad_accum):
      x0 = torch.tensor([next(stream) for _ in range(B)], device=device)
      consumed += B
      kl_term = torch.zeros((), device=device)
      if lc.bits:
        z, p = latent.bernoulli_st(adapter.encoder(x0).float())
        kl_bits = latent.kl_to_uniform(p)
        adapter.set_z(z)
        per_bit = kl_bits.mean(0)
        if lc.free_bits > 0:
          per_bit = per_bit.clamp(min=lc.free_bits)
        kl_term = per_bit.sum()
        run['kl'] += kl_bits.sum(-1).mean().item()
      nll = model._forward_pass_diffusion(x0).sum(-1).mean()
      adapter.set_z(None)
      ((nll + beta * kl_term) / (L * lc.grad_accum)).backward()
      run['nll'] += nll.item() / L
      run['n'] += 1
    for g in groups:   # clip separately so the large adaLN gradients don't
      torch.nn.utils.clip_grad_norm_(g['params'], 1.0)   # shrink the latent's
    opt.step()
    step += 1
    if step % lc.log_every == 0:
      n = run['n']
      logger.info(f'step {step:>5}/{lc.steps}  nll/tok {run["nll"] / n:.4f}  '
                  f'kl {run["kl"] / n:.2f} nats/seq  beta {beta:.3f}  '
                  f'{(time.time() - t0) / lc.log_every:.2f} s/step')
      t0, run = time.time(), {'nll': 0.0, 'kl': 0.0, 'n': 0}
    if step % lc.eval_every == 0 or step == lc.steps:
      m = evaluate()
      history.append(m)
      log_eval(m)
      save(latest, with_opt=True)
      with open(metrics_path, 'w') as f:
        json.dump(history, f, indent=1)
  final = os.path.join(lc.save_dir, 'final.pt')
  save(final, with_opt=False)
  logger.info(f'Done. Adapter for sampling: latent.ckpt_path={final}')


def _ppl_eval(config, logger, tokenizer):
  logger.info('Starting Zero Shot Eval.')

  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=config.seed)
  trainer.validate(model, valid_ds)


def _train(config, logger, tokenizer):
  logger.info('Starting Training.')
  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  train_ds, valid_ds = dataloader.get_dataloaders(
    config, tokenizer)
  _print_batch(train_ds, valid_ds, tokenizer)

  model = diffusion.Diffusion(
    config, tokenizer=valid_ds.tokenizer)

  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)

  if config.mode == 'sample_eval':
    generate_samples(config, logger, tokenizer)
  elif config.mode == 'latent_finetune':
    latent_finetune(config, logger, tokenizer)
  elif config.mode == 'revealer_label':
    revealer_label(config, logger, tokenizer)
  elif config.mode == 'revealer_train':
    revealer_train(config, logger, tokenizer)
  elif config.mode == 'sample_sweep':
    sample_sweep(config, logger, tokenizer)
  elif config.mode == 'ppl_eval':
    _ppl_eval(config, logger, tokenizer)
  else:
    _train(config, logger, tokenizer)


if __name__ == '__main__':
  main()