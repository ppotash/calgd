# Running CALGD on Google Colab (T4 or L4)

This fork runs MDLM evaluation, sampling, revealer training and the latent probe on Colab. A free
T4 works for everything; an L4 (paid compute units) is about 2x faster with identical settings
and numerics. Stock MDLM assumes an
Ampere-or-newer GPU with flash-attention, and Colab's Python 3.13 / numpy 2 environment
conflicts with the repo's pinned dependencies, so a few workarounds are needed:

- **No flash-attention on a T4.** The repo's flash-attn and mamba imports are optional in this
  fork, and evaluation uses the `kuleshov-group/mdlm-no_flashattn-fp32-owt` checkpoint. That
  checkpoint still calls flash-attn for rotary embeddings, so the setup patches in a pure-PyTorch
  equivalent (checked against flash-attn's documented formula).
- **`datasets==2.18` needs numpy 1.x**, which has no prebuilt wheel for Python 3.13. It is built
  from source once (~6 min) and the wheel is cached on Drive.
- **Colab's JAX requires numpy 2** and is imported by `transformers` if present, so it is
  uninstalled, and TensorFlow/Flax imports are disabled with environment variables.

## Layout

| Location | Contents | Survives a runtime reset? |
|---|---|---|
| `/content/calgd` | this repo (cloned each session) | no, so push changes to GitHub |
| `/content/mdlm-nofa` | patched checkpoint (downloaded each session) | no |
| `MyDrive/calgd/data` | dataset cache | yes |
| `MyDrive/calgd/outputs` | Hydra run dirs, `sample_sweep.json` results | yes |
| `MyDrive/calgd/wheels` | cached numpy 1.26.4 wheel | yes |

One-time setup: create the `MyDrive/calgd` folder, and add a GitHub token as a Colab secret named
`GITHUB_TOKEN` (fine-grained token with **Contents: Read and write** on this repo), with
"Notebook access" enabled.

## Setup

After a fresh runtime, run cells A and B, restart the session, then run cell C.
After a plain restart (Runtime → Restart session), only cell C is needed.

**Cell A: Drive and repo**
```python
import os
from google.colab import drive
drive.mount('/content/drive')
if not os.path.exists('/content/calgd'):
    !git clone https://github.com/ppotash/calgd.git /content/calgd
%cd /content/calgd
!mkdir -p /content/drive/MyDrive/calgd/data /content/drive/MyDrive/calgd/outputs
!git log --oneline -3
```

**Cell B: dependencies** (then Runtime → Restart session)
```python
W = "/content/drive/MyDrive/calgd/wheels"
!mkdir -p {W}
!ls {W} | grep -q "numpy-1.26.4" || pip wheel -q --no-deps "numpy==1.26.4" -w {W}
!pip install -q --find-links {W} "numpy==1.26.4" "transformers==4.38.2" "lightning==2.2.1" \
  "hydra-core==1.3.2" "omegaconf==2.3.0" "datasets==2.18.0" "fsspec==2024.2.0" \
  einops timm rich torchmetrics "setuptools<70"
!pip uninstall -y -q jax jaxlib flax optax chex orbax-checkpoint
```
pip prints dependency-conflict warnings about Colab's preinstalled packages; they can be ignored.

**Cell C: environment flags and patched checkpoint** (after every restart)
```python
import os
os.environ["USE_TF"] = "0"; os.environ["USE_FLAX"] = "0"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
%cd /content/calgd

if not os.path.exists("/content/mdlm-nofa"):
    from huggingface_hub import snapshot_download
    snapshot_download("kuleshov-group/mdlm-no_flashattn-fp32-owt", local_dir="/content/mdlm-nofa")
    p = "/content/mdlm-nofa/modeling_mdlm_2.py"
    s = open(p).read()
    imp = "import flash_attn\nimport flash_attn.layers.rotary\n"
    call = "return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)"
    assert s.count(imp) == 1 and s.count(call) == 1
    s = s.replace(imp, "try:\n  import flash_attn\n  import flash_attn.layers.rotary\nexcept ImportError:\n  flash_attn = None\n")
    s = s.replace(call, "return _rotary_qkv_torch(qkv, cos, sin)")
    s += '''

def _rotary_qkv_torch(qkv, cos, sin):
  # Pure-PyTorch equivalent of flash_attn apply_rotary_emb_qkv_ (non-interleaved).
  rd = cos.shape[-1] * 2
  cos = torch.cat((cos, cos), dim=-1)[None, :, None, None, :]
  sin = torch.cat((sin, sin), dim=-1)[None, :, None, None, :]
  qk = qkv[:, :, :2, :, :rd]
  half = rd // 2
  rot = torch.cat((-qk[..., half:], qk[..., :half]), dim=-1)
  out = qkv.clone()
  out[:, :, :2, :, :rd] = qk * cos + rot * sin
  return out
'''
    open(p, "w").write(s)
    print("checkpoint downloaded and patched")

import numpy, torch, transformers, datasets, lightning
print(numpy.__version__, transformers.__version__, datasets.__version__, lightning.__version__,
      torch.cuda.get_device_name(0))
# expected: 1.26.4 4.38.2 2.18.0 2.2.1 Tesla T4
```

## Running

T4 settings: the no-flash checkpoint, `trainer.precision=32`, and batch size 8. Always pass
`hydra.run.dir` on Drive so results survive a reset (`\$` stops the shell expanding `${now:...}`).

**Perplexity (zero-shot)**
```python
!python main.py mode=ppl_eval backbone=hf_dit eval.checkpoint_path=/content/mdlm-nofa \
  data=wikitext2 data.cache_dir=/content/drive/MyDrive/calgd/data \
  model.length=1024 loader.batch_size=8 loader.eval_batch_size=8 \
  trainer.precision=32 eval.generate_samples=False wandb=null \
  hydra.run.dir=/content/drive/MyDrive/calgd/outputs/ppl/\${now:%Y.%m.%d}/\${now:%H%M%S}
```
Use `data=ptb` for PTB and `seed=N` for repeat runs.

**Sample sweep (gen-PPL, entropy and time vs. number of steps)**
```python
!python main.py mode=sample_sweep backbone=hf_dit eval.checkpoint_path=/content/mdlm-nofa eval.disable_ema=True \
  data=openwebtext-split model.length=1024 \
  sampling.predictor=ordered sampling.order=confidence \
  loader.eval_batch_size=8 sampling.num_sample_batches=16 wandb=null \
  hydra.run.dir=/content/drive/MyDrive/calgd/outputs/sweeps/\${now:%Y.%m.%d}/\${now:%H%M%S}
```
Samplers: `sampling.predictor=ddpm_cache` (MDLM default), or `predictor=ordered` with
`sampling.order=random|confidence`. Step counts come from `sampling.sweep_steps` (default
`[8,16,32,64,128]`; override as `'sampling.sweep_steps=[8,16]'`). Sampling uses float64 by
default; pass `sampling.fp64=False` to reproduce stock MDLM behaviour.

**Revealer** (label once, then train the head in seconds; labeling resumes after a disconnect)
```python
common = ("backbone=hf_dit eval.checkpoint_path=/content/mdlm-nofa eval.disable_ema=True "
          "data=openwebtext-split model.length=1024 wandb=null")
lab = "/content/drive/MyDrive/calgd/revealer/labels_v2"
!python main.py mode=revealer_label {common} revealer.label_dir={lab} revealer.n_shards=10 revealer.n_samples=4
!python main.py mode=revealer_train {common} revealer.label_dir={lab} revealer.head_path={lab}/head_sampled.pt
```
Sample with a trained head: `sampling.predictor=ordered sampling.order=revealer
sampling.revealer_path=<head.pt> sampling.revealer_temp=<T>`.

**Latent-bits probe** (standalone; results append to `results.json`, finished runs are skipped)
```python
out = "/content/drive/MyDrive/calgd/probe_v1"
!python probes/latent_probe.py --out {out} --bits 0 --tag elbo
!python probes/latent_probe.py --out {out} --bits 4 --beta 0.1 --tag beta0.1
```

**Long runs and disconnects.** Hydra outputs and all results go to Drive, and the sweep, labeling
and probe commands all skip or resume finished work, so after a reset rerun the setup cells and
then the same command. For loops over several configurations, run them through `subprocess` and
print the last lines on failure; a filtered `!python ... | grep` hides errors such as "can't open
file" after a restart reset the working directory.

## Results

Baseline numbers and all experiment results are in the main [README](../README.md).

## Pushing changes

Code in `/content/calgd` is lost on a runtime reset, so commit and push before disconnecting:
```python
import subprocess
from google.colab import userdata
token = userdata.get("GITHUB_TOKEN")
r = subprocess.run(["git", "push", f"https://ppotash:{token}@github.com/ppotash/calgd.git", "HEAD:master"],
                   capture_output=True, text=True)
print((r.stdout + r.stderr).replace(token, "***"))   # never print the token
!git fetch -q origin && git status -sb | head -1
```
Pushing to the URL keeps the token out of `.git/config`, but it does not update the local
`origin/master` ref; the `git fetch` refreshes it so `git status` shows the true state.

## Gotchas

- **Restart vs. reset.** Runtime → Restart session keeps files and packages; a disconnect or
  "Disconnect and delete runtime" wipes everything outside Drive. Idle sessions are dropped on the
  free tier, but a running cell counts as activity.
- **Use `%cd`, not `!cd`.** `!cd` runs in a throwaway shell and changes nothing. A session
  restart also resets the working directory to `/content`, so rerun cell C.
- **numpy `dtype size changed` errors** mean packages were imported in the same session that
  reinstalled numpy: restart the session after cell B.
- **Uploaded patch files may get Windows line endings.** Fix them before `git apply` with
  `!sed -i 's/\r$//' file.patch && echo >> file.patch`.
- **Hydra changes the working directory** into the run dir, so use absolute paths in overrides.
