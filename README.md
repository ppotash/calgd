# CALGD: Context-Aware Latent-Gated Diffusion

**Can a sampled discrete latent, or an information-gain revealer, improve few-step sampling in
masked diffusion language models?**

This is a research fork of [MDLM](https://github.com/kuleshov-group/mdlm) (Sahoo et al.,
NeurIPS 2024). All credit for the base framework, models and checkpoints goes to the original
authors; see [Based on MDLM](#based-on-mdlm) and [Citation](#citation).

> **Status:** the revealer is a documented **negative** result. Latent bits give large gains on
> synthetic data, do **not** work when fine-tuned into a pretrained MDLM, and, when trained jointly
> from scratch on real text with prefix-token conditioning and a learned prior, improve few-step
> samples by **12–14% at 2–4 steps and 6–9% at 8–16 steps** (2 seeds, small models). Next: does
> the benefit grow with model quality? See [Roadmap](#roadmap).

## Key findings

1. **Unmasking order matters a lot, but a learned "information-gain" order adds nothing over
   confidence.** Measured against the true text, the most informative reveal is mostly the one
   most likely to be correct, which confidence already captures.
2. **A sequence-level latent collapses under the exact likelihood bound.** A masked diffusion
   model's bound is already tight without it; the latent's value is only in few-step sampling,
   which the per-token training loss never measures. Training needs a down-weighted KL or free bits.
3. **On synthetic data with hidden global modes, the latent is a large win** (one-step validity
   0 → 0.69), at no cost in likelihood.
4. **Fine-tuned into a pretrained MDLM, the latent barely helps,** even when its bits carry
   guaranteed-useful oracle information: a strong denoiser infers document-level properties from
   context, and a model pretrained with a constant conditioning vector does not readily use a
   variable one.
5. **Trained jointly from scratch on real text it helps modestly,** but only with three ingredients:
   latent dropout (so the model still learns context), prefix-token conditioning (so z does not
   interfere with token representations) and a learned prior over codes. The gain is far smaller
   than on synthetic data because most of a text model's few-step errors are local, which a
   document-level code cannot fix.

## Motivation

Masked diffusion LMs generate by unmasking many tokens in parallel. Tokens unmasked in the same
step are sampled independently given the current context, which is the main source of quality
loss when sampling with few steps. This project tests two ways of reducing that loss:

1. **Latent bits.** A short vector of discrete bits `z` is sampled once per sequence and
   conditions the denoiser, so tokens decoded in the same step share `z` and can be correlated
   instead of independent. Training uses an encoder `q(z | x0)` and a KL term to a prior; the
   working recipe on real text gives `z` to the denoiser as prefix tokens, drops it for half of
   training sequences, and samples it from a prior fitted to the encoder's codes. Reported bounds
   always use the full KL.
2. **Information-gain revealer.** A small head scores masked positions by how much revealing them
   would reduce the loss on the rest of the sequence, and the sampler unmasks the highest-scoring
   positions first. It is trained on counterfactual reveals with a pairwise ranking loss.

The claim being tested is about **sample quality at a fixed small number of steps**; neither idea
is expected to improve likelihood much.

## Evaluation protocol

- **Sample quality:** generative perplexity under GPT-2 Large **and** per-sample token entropy
  (plus repeated 4-grams for the from-scratch models). Lower generative perplexity can come from
  repetitive text, so samplers are compared at matched entropy where possible.
- **Sampler baselines:** MDLM's `ddpm_cache`; fixed-count unmasking in random order; and
  confidence order with MaskGIT-style annealed noise, swept over temperature to trace a curve.
- **Latent models:** compared with the *same weights* with `z` switched off (and with a matched
  control trained without a latent), with bootstrap confidence intervals over samples.
- **Likelihood:** validation bounds with paired randomness, so differences between runs and steps
  are exact comparisons.
- **Noise:** single-run perplexity varies by about ±4% on WikiText-2 and generative perplexity
  from 64 samples by about ±10%, so comparisons use several seeds or confidence intervals.

## Roadmap

- [x] **Reproduce the baseline** on the released MDLM checkpoint.
- [x] **Evaluation pipeline:** `mode=sample_sweep`, sample entropy, ordered samplers, float64
      sampling.
- [x] **Baseline sweeps** for MDLM, random and confidence ordering.
- [x] **Revealer on the frozen MDLM checkpoint** (negative).
- [x] **Synthetic probes** for the latent (positive, 3 seeds).
- [x] **Latent fine-tune of MDLM** (negative, with oracle diagnostics).
- [x] **Latent trained jointly from scratch on real text, small scale** (positive but modest,
      2 seeds).
- [ ] **Scale up** (~125M parameters, several billion tokens): does the few-step benefit grow as
      the model makes fewer local errors?
- [ ] **Mixed-domain corpus** (English, code, other languages): data with stronger global modes,
      closer to the synthetic setting.

## Results

### Baseline reproduction

Released checkpoint `kuleshov-group/mdlm-no_flashattn-fp32-owt`, evaluated on a Colab T4:

| Metric | Value | Notes |
|---|---|---|
| WikiText-2 val PPL | 35.2 (range 33.4–36.4) | 3 seeds; paper reports ≤32.83 |
| PTB val PPL | ~108 | 1 seed; paper reports ≤95.26 |

The paper's zero-shot numbers come from a longer-trained model, which likely explains most of the
gap. All comparisons in this project are against these measured baselines.

### Sampler baselines (64 samples per point)

Gen-PPL under GPT-2 Large; real OpenWebText entropy is 5.45 ± 0.16 nats (1,024-token chunks).

| Steps | `ddpm_cache` | Random order | Confidence @ entropy 5.45 |
|---|---|---|---|
| 16 | 342 | 348 | ≈117 |
| 32 | 196 | 184 | ≈75 |
| 64 | 137 | 140 | ≈60 |

- **Ordering matters a lot:** at real-text entropy, confidence ordering lowers generative
  perplexity 2.3–3× versus random order, and at 16 steps it beats random order at 64.
- **Pure confidence ordering collapses** into repeated filler tokens (entropy 0.3–1.4); annealed
  noise fixes this, and the temperature trades fluency for diversity (T≈20 reaches real-text
  entropy).
- **Random order and MDLM's sampler are equivalent:** only which positions are revealed matters,
  not whether the count per step is fixed.

![Sampler baselines and revealer](docs/baseline_frontier.png)

### Revealer: negative result

A reveal head on the frozen MDLM checkpoint, trained on counterfactual labels: for 8 masked
candidates per sequence, reveal the candidate with a value sampled from the model (4 independent
samples) and measure the drop in mean loss on the remaining masked tokens. Pairwise ranking loss;
the head learns a correction on top of confidence. 2,581 labeled sequences (2,063 train / 518 val).

- **Labels are dominated by whether the sampled value is right.** Two halves of the samples agree
  on 73% of candidate pairs (within-sequence r = 0.42; ≈0.59 for the 4-sample average).
- **Ranking:** head 0.600 vs. confidence 0.592 pairwise accuracy (+0.009, 95% CI +0.002 to +0.016).
  On sample-averaged gains, head 0.633 vs. 0.627 for Σp² (mean sampled-token probability).
- **Sampling:** at matched temperature the revealer is indistinguishable from confidence ordering
  (gen-PPL within ~2% at 16/32/64 steps, T=10 and T=20; diamonds in the figure above).

Interpretation: measured against the true text, "most informative reveal" collapses to "most likely
correct", which confidence already captures. At the noise level needed for real-text diversity
(T≈20), early unmasking is close to random for any scorer, so a scorer would need to be far better
than confidence to matter. Cost: about 3 L4-hours.

### Latent-bits probe (synthetic data)

Sequences of 32 tokens generated by one of 16 hidden modes; a sample is valid if all its tokens
come from a single mode. Small masked diffusion transformers trained from scratch, with and
without 4 latent bits (encoder in training, uniform prior at sampling). 3 seeds; each seed also
draws a different set of modes.

| Valid samples (mean of 3 seeds) | 1 step | 4 steps | 16 steps | 64 steps |
|---|---|---|---|---|
| No latent | 0.00 | 0.01 | 0.35 | 0.70 |
| 4 bits, KL weight 0.1 | 0.69 | 0.76 | 0.87 | 0.89 |

- **Trained on the exact likelihood bound, the latent collapses** (KL → 0.01–0.03 nats) and
  gives no benefit. With a perfect denoiser a masked diffusion model's bound is already tight,
  so a latent's KL cost cancels its gain; its value is only in few-step sampling, which the
  per-token training loss never measures.
- **With KL weight 0.1 (or free bits ≈ 0.65 nats/bit) the code is used and sharp,** and
  few-step validity improves dramatically, while the bound (evaluated with the full KL) is
  slightly better than without the latent in every seed. Mode coverage stays near uniform, and
  samples' true likelihood stays above the data's own, so the gain is not bought with diversity.
- **Code sharpness matters more than size.** A noisy code (free bits 0.5) barely helps at one
  step; 8 bits for 16 modes wastes most of the prior on codes the encoder never uses.
- **On Markov-structured data,** the latent removes mode-mixing errors (one-step local validity
  0.22 → 0.44) but cannot replace steps for within-mode dependencies.

![Latent probe](docs/probe/probe_v1.png)

Script: `probes/latent_probe.py`; per-run figures and metrics in `docs/probe/`.

### Latent bits on a pretrained MDLM (fine-tuning): negative result

Latent bits added to the released checkpoint by fine-tuning: an encoder q(z|x0), a projection
of z added to the denoiser's conditioning vector (zero-initialised, so training starts exactly
at the released model), trained with MDLM's loss plus a down-weighted KL. Every run has a matched
control trained identically without a latent. Bounds are nats/token on 128 held-out sequences,
with paired randomness so differences between runs are exact comparisons.

| Setup (A100, 1,000–2,000 steps) | Uses z? (loss gap, random vs. true bits) | vs. matched control | Samples: word present when bit on / off |
|---|---|---|---|
| Learned encoder, 16 bits, KL weight 0.1 / 0 / free bits | KL collapses to ~0; no gap | equal | — |
| Frozen random-hash encoder, 16 bits | no gap (true bits slightly worse) | slightly worse | — |
| Oracle: 64 bits = "word w occurs", adaLN layers only | 0.0028 | worse by 0.006 | 0.55 / 0.54 |
| Oracle, + z added to token embeddings | 0.0108 | worse by 0.027 | 0.58 / 0.50 |
| Oracle, full fine-tune | 0.0028 | worse by 0.006 | 0.59 / 0.52 |
| Oracle, full fine-tune, trained only on ≥80% masked inputs | 0.0075 (on that range) | worse by 0.006 | 0.53 / 0.45 (16 steps), 0.59 / 0.48 (64) |

- **Information budget.** A latent can lower the loss by at most the information it carries:
  16 bits ≤ ~11 nats per 1,024-token sequence (≤0.011 nats/token), 64 bits ≤ 0.043. Loss gaps
  are therefore a low-power test; sample steering is the more sensitive one.
- **The denoiser picks up only a sliver of even guaranteed-useful information,** and conditioning
  costs more than it gains: with the *true* oracle bits, every configuration is slightly worse
  than its control. More trainable parameters (full fine-tune) or concentrating training where
  the bits matter (heavily masked inputs) did not change this.
- **Interpretation:** a strong pretrained denoiser already infers document-level properties from
  visible text, so a latent adds information only when nearly everything is masked, and a model
  pretrained with a constant conditioning vector does not readily learn to use a variable one.
- **Caveat:** fine-tuning itself degrades sampling (gen-PPL ≈ 470–490 / 205–218 at 16 / 64 steps
  for all fine-tuned models, latent or not, vs. 343 / 140 for the released checkpoint on the same
  setup) while slightly improving the bound. Likely cause, untested: the released weights are an
  EMA; the fine-tunes save raw weights. This does not affect the matched comparisons above.

Code: `latent.py`; `mode=latent_finetune` (options: `latent.oracle`, `latent.train_scope`,
`latent.t_min`, `latent.input_inject`, `latent.shuffle_stream`); `sample_sweep` with
`latent.ckpt_path` and `latent.sample_z`.

### Latent bits trained jointly from scratch on real text

A 51–53M-parameter masked diffusion transformer (8 layers, width 512) trained from scratch on
256-token windows of OpenWebText (500M tokens, GPT-2 tokenizer; 15k steps × 128 sequences,
≈35 min per run on an A100, EMA weights for evaluation). Each latent run has a matched control
with identical data order and trunk initialisation. Script: `probes/text_scratch.py`.

**What it took.** Five configurations failed before one worked:

| Variant | Outcome |
|---|---|
| z added to every token embedding | latent used, but training stalls on the unigram plateau ~4,000 steps longer; ends 0.68 nats/token behind the control |
| + encoder no longer trains the shared embedding | same stall: the latent itself is the shortcut |
| + latent dropout 0.5 | learns normally, but the latent collapses as context is learned |
| + free bits (0.3 nats/bit) | still collapses: any z input makes predictions *worse* than none while context is being learned, so the denoiser learns to ignore it |
| **z as 4 prefix tokens + dropout 0.5 + free bits** | **learns normally and keeps using z** |

**Likelihood (step 15k).** Within the latent model, the document's own code beats z switched off
by 0.011 / 0.012 nats/token (seeds 0 / 1) and a random code by 0.029 / 0.026; the gaps are stable
over the last 10k steps. With a learned autoregressive prior over codes (texts use ~17k of the
65,536 codes; prior NLL 9.1 vs. 11.1 nats for uniform), the KL falls from 5.4 to 3.5 nats/sequence
and the bound comes within 0.001–0.003 nats/token of the same model with z off: the latent is
nearly free in likelihood but does not improve it.

**Few-step samples** (512 samples; gen-PPL relative to the same model with z off; z from the
learned prior):

| Steps | Seed 0 | Seed 1 | Combined (95% CI) |
|---|---|---|---|
| 2 | 0.89 | 0.86 | **0.875** (0.852–0.898) |
| 4 | 0.87 | 0.86 | **0.864** (0.839–0.891) |
| 8 | 0.93 | 0.89 | **0.913** (0.882–0.943) |
| 16 | 0.95 | 0.92 | **0.936** (0.903–0.967) |

The combined interval reflects sampling noise within each model; with only two training seeds
it does not capture seed-to-seed variation, though the two seeds agree closely.

The control matches the latent model with z off (ratios 0.98–1.04), so the gain is not from the
prefix tokens making a better model. The benefit is largest at 2–4 steps and fades with more
steps, as expected if the latent fixes errors from committing many tokens at once. Drawing z
uniformly instead of from the learned prior loses most of the gain beyond 4 steps.

**Caveats.** Two training seeds, small models: all samples are weak in absolute terms (gen-PPL
≈2,400 at 2 steps and ≈650 at 16, vs. 19.6 for real text). Samples with z are 0.01–0.04 nats lower
in entropy, which accounts for part of the gain.

**Interpretation.** A sequence-level latent helps real-text few-step sampling only modestly
(≈10%), compared with the synthetic probe, because most of a text model's uncertainty and few-step
errors are local, which a document-level code cannot fix, and because context recovers most
document-level information once a fraction of tokens is visible. Whether the benefit grows as
stronger models make fewer local errors is the open question for scaling up.

Per-run training histories and sample metrics: `docs/scratch/`.

## Changes from upstream MDLM

- **Optional heavy dependencies.** `flash-attn`, `mamba-ssm` and `causal-conv1d` imports are
  optional, so evaluation runs on GPUs without flash-attention (e.g. a T4).
- **`mode=sample_sweep`.** Generates samples at several step counts and records generative
  perplexity, entropy, seconds per sample and the samples themselves to `sample_sweep.json`.
- **`sampling.predictor=ordered`.** Fixed-count unmasking with `sampling.order=` `random`,
  `confidence` (with `sampling.confidence_temp`, MaskGIT-style annealed noise) or `revealer`
  (with `sampling.revealer_path` and `sampling.revealer_temp`).
- **`sampling.fp64` (default `True`).** Float64 categorical sampling, avoiding float32 truncation
  that understates generative perplexity. Set `False` to reproduce stock MDLM sampling.
- **Revealer pipeline** (`revealer.py`): `mode=revealer_label` (counterfactual information-gain
  labels, resumable) and `mode=revealer_train` (head training with confidence baselines,
  reliability and bootstrap confidence intervals).
- **Latent fine-tuning** (`latent.py`): `mode=latent_finetune` with a matched-control option
  (`latent.bits=0`), oracle bits, full or adaLN-only fine-tuning, high-noise training, and latent
  sampling in `sample_sweep` (`latent.ckpt_path`, `latent.sample_z`).
- **Latent probe** (`probes/latent_probe.py`): standalone synthetic experiment.
- **From-scratch text experiment** (`probes/text_scratch.py`): data preparation, two-arm training,
  learned prior fitting, few-step sampling with bootstrap-ready per-sample scores.
- **GPT-2 Large is loaded once** for generative perplexity instead of on every batch.

## Quickstart

On Colab, follow **[docs/colab.md](docs/colab.md)** for setup (Python 3.13 and flash-attention
workarounds) and for the revealer and probe commands. For example:

```bash
# Zero-shot perplexity
python main.py mode=ppl_eval backbone=hf_dit eval.checkpoint_path=/content/mdlm-nofa \
  data=wikitext2 model.length=1024 loader.batch_size=8 loader.eval_batch_size=8 \
  trainer.precision=32 eval.generate_samples=False wandb=null

# Sample sweep with confidence-ordered unmasking
python main.py mode=sample_sweep backbone=hf_dit eval.checkpoint_path=/content/mdlm-nofa \
  eval.disable_ema=True data=openwebtext-split model.length=1024 \
  sampling.predictor=ordered sampling.order=confidence sampling.confidence_temp=20 \
  loader.eval_batch_size=8 sampling.num_sample_batches=8 wandb=null

# Latent probe (synthetic)
python probes/latent_probe.py --out probe_out --bits 0 4 --beta 0.1 --tag beta0.1
```

The from-scratch text experiment needs no MDLM setup, only PyTorch, `transformers` and `datasets`
(a fresh Colab runtime has them):

```bash
D=/path/on/drive/scratch
python probes/text_scratch.py prepare   --out $D --tokens 500000000   # ~6 min, CPU is fine
python probes/text_scratch.py train     --out $D --bits 0              # control
python probes/text_scratch.py train     --out $D --bits 16 --z_mode prefix --z_dropout 0.5 --free_bits 0.3 --tag v6
python probes/text_scratch.py fit_prior --out $D --bits 16 --tag v6
python probes/text_scratch.py sample    --out $D --bits 0 --n_samples 512 --sample_steps 2 4 8 16
python probes/text_scratch.py sample    --out $D --bits 16 --tag v6 --n_samples 512 --sample_steps 2 4 8 16 --z_from learned
python probes/text_scratch.py sample    --out $D --bits 16 --tag v6 --n_samples 512 --sample_steps 2 4 8 16 --null_z
python probes/text_scratch.py compare   --out $D
```

On Colab, call `drive.flush_and_unmount()` before closing a runtime that wrote large files to
Drive; otherwise they may never finish uploading.

On a GPU with flash-attention (Ampere or newer), `eval.checkpoint_path=kuleshov-group/mdlm-owt`
also works for the MDLM commands; drop `trainer.precision=32`.

## Repository layout

| Path | Contents |
|---|---|
| `main.py` | Entry point: training, `ppl_eval`, `sample_eval`, `sample_sweep`, `revealer_label`, `revealer_train`, `latent_finetune` |
| `diffusion.py` | Forward/reverse diffusion, samplers (incl. `ordered`), generative perplexity |
| `revealer.py` | Revealer head, information-gain labels, ranking loss, metrics |
| `latent.py` | Latent adapter for a pretrained MDLM: encoder, oracle bits, conditioning hooks, checkpoints |
| `probes/latent_probe.py` | Synthetic latent-bits probe |
| `probes/text_scratch.py` | From-scratch latent experiment on real text |
| `noise_schedule.py` | Noise schedules |
| `dataloader.py` | Datasets and dataloaders |
| `models/` | Denoisers: DiT, AR transformer, Mamba |
| `configs/` | Hydra configs (data, models, noise, sampling, revealer, latent) |
| `scripts/` | Upstream Slurm scripts |
| `docs/` | Colab guide, figures, probe and from-scratch results |

## Based on MDLM

This repository is a fork of [kuleshov-group/mdlm](https://github.com/kuleshov-group/mdlm), which
introduced MDLM, a masked diffusion LM with a substitution-based parameterization that reduces the
absorbing-state diffusion loss to a mixture of masked-language-modeling losses. The upstream
authors note that an improved implementation is available in the
[DUO repo](https://github.com/s-sahoo/duo), and MDMs with KV caching in the
[Eso-LMs repo](https://github.com/s-sahoo/Eso-LMs).

<details>
<summary>Upstream usage reference</summary>

**Checkpoints.** MDLM trained on OpenWebText for 1M steps:
[kuleshov-group/mdlm-owt](https://huggingface.co/kuleshov-group/mdlm-owt) (flash-attention) and
[kuleshov-group/mdlm-no_flashattn-fp32-owt](https://huggingface.co/kuleshov-group/mdlm-no_flashattn-fp32-owt)
(regular attention, float32). AR and SEDD baseline checkpoints are in the upstream
[Google Drive folder](https://drive.google.com/drive/folders/16LuuptK7Xfk-vzhQYZBZ0SA-B-BFluau?usp=sharing).

**Environment (upstream).** `conda env create -f requirements.yaml && conda activate mdlm`.

**Samplers.** `sampling.predictor` takes `ddpm_cache` (MDLM's fast sampler), `ddpm` (D3PM
ancestral sampling), `analytic` (SEDD), and in this fork `ordered`. Semi-autoregressive generation
of longer sequences: `sampling.semi_ar=True sampling.stride_length=512 sampling.num_strides=2`.

**Generate samples:**
```bash
python main.py mode=sample_eval eval.checkpoint_path=kuleshov-group/mdlm-owt \
  data=openwebtext-split model.length=1024 sampling.predictor=ddpm_cache \
  sampling.steps=1000 loader.eval_batch_size=1 sampling.num_sample_batches=10 backbone=hf_dit
```

**Train MDLM from scratch on OpenWebText:**
```bash
python main.py model=small data=openwebtext-split wandb.name=mdlm-owt parameterization=subs \
  model.length=1024 eval.compute_generative_perplexity=True sampling.steps=1000
```
`loader.batch_size` and `loader.eval_batch_size` set the per-GPU batch sizes; Lightning uses
gradient accumulation to reach the global batch size. Slurm scripts are in `scripts/`.

**Perplexity of a local checkpoint:**
```bash
python main.py mode=ppl_eval loader.batch_size=16 loader.eval_batch_size=16 \
  data=openwebtext-split model=small parameterization=subs backbone=dit model.length=1024 \
  eval.checkpoint_path=/path/to/checkpoint/mdlm.ckpt +wandb.offline=true
```
For the AR baseline use `model=small-ar parameterization=ar backbone=ar`; for SEDD use
`parameterization=sedd time_conditioning=True sampling.predictor=analytic`.

</details>

## Citation

If you use this code, please cite MDLM:

```bibtex
@inproceedings{
sahoo2024simple,
title={Simple and Effective Masked Diffusion Language Models},
author={Subham Sekhar Sahoo and Marianne Arriola and Aaron Gokaslan and Edgar Mariano Marroquin and Alexander M Rush and Yair Schiff and Justin T Chiu and Volodymyr Kuleshov},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=L4uaAR4ArM}
}
```

## Acknowledgements and license

Built on [MDLM](https://github.com/kuleshov-group/mdlm), which itself builds on
[SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion). Licensed under Apache-2.0,
as is the upstream project; see `LICENSE`.
