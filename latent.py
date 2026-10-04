"""Latent bits for a pretrained MDLM denoiser.

A sequence-level code z (K Bernoulli bits) conditions the denoiser through its
existing conditioning path: MDLM's DiT computes c = SiLU(sigma_map(sigma)) and
feeds c to every block's adaLN modulation. A forward hook adds z_proj(z) to the
sigma_map output, so z modulates every layer. z_proj starts at zero, so an
untrained adapter leaves the pretrained model exactly unchanged.

Training (mode=latent_finetune): encoder q(z | x0) over the clean sequence
(straight-through Bernoulli), MDLM's diffusion loss given z, plus a KL to the
uniform prior with weight latent.beta < 1. (At beta = 1 the latent collapses: a
masked diffusion model's bound is already tight without it; see the probe.)
Reported bounds always use the full KL.

Sampling: z ~ Bernoulli(0.5)^K once per sequence, held fixed for all steps.

latent.bits = 0 gives the matched control: same trainable parameters (the adaLN
modulation layers), same data and steps, no latent.
"""
import torch
import torch.nn as nn


def _find(backbone, suffix):
  mods = [(n, m) for n, m in backbone.named_modules()
          if n.split('.')[-1] == suffix]
  if len(mods) != 1:
    tail = [n for n, _ in backbone.named_modules()][-30:]
    raise RuntimeError(f'Expected one module named {suffix!r}, found '
                       f'{[n for n, _ in mods]}. Last modules: {tail}')
  return mods[0]


def adaln_parameters(backbone):
  """(name, param) for every adaLN modulation layer (blocks + final layer)."""
  params = [(n, p) for n, p in backbone.named_parameters()
            if 'adaLN_modulation' in n]
  if not params:
    names = [n for n, _ in backbone.named_parameters()][:40]
    raise RuntimeError(f'No adaLN_modulation parameters found. First: {names}')
  return params


def vocab_embedding(backbone, vocab_size):
  """The backbone's token embedding matrix [V, d], if it can be found."""
  for n, p in backbone.named_parameters():
    if 'vocab_embed' in n and p.ndim == 2 and p.shape[0] == vocab_size:
      return p.detach()
  return None


class SeqEncoder(nn.Module):
  """q(z | x0): small transformer over the clean sequence -> K bit logits.
  Token embeddings are a frozen copy of the denoiser's, projected down."""

  def __init__(self, vocab_size, seq_len, bits, dim=256, layers=2, heads=4,
               emb_init=None, emb_dim=768):
    super().__init__()
    if emb_init is not None:
      emb_dim = emb_init.shape[1]
      self.register_buffer('emb', emb_init.clone().float())
    else:
      self.register_buffer('emb', torch.randn(vocab_size, emb_dim) * 0.02)
    self.proj = nn.Linear(emb_dim, dim)
    self.pos = nn.Parameter(torch.randn(seq_len, dim) * 0.02)
    layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0,
                                       activation='gelu', batch_first=True,
                                       norm_first=True)
    self.body = nn.TransformerEncoder(layer, layers,
                                      enable_nested_tensor=False)
    self.norm = nn.LayerNorm(dim)
    self.head = nn.Linear(dim, bits)
    nn.init.zeros_(self.head.weight)   # starts at p = 0.5 for every bit
    nn.init.zeros_(self.head.bias)

  def forward(self, x0):
    h = self.proj(self.emb[x0]) + self.pos[:x0.shape[1]]
    return self.head(self.norm(self.body(h)).mean(1))


def bernoulli_st(logits):
  p = torch.sigmoid(logits)
  hard = torch.bernoulli(p.detach())
  return hard + p - p.detach(), p


def kl_to_uniform(p):
  """Per-bit KL(Bern(p) || Bern(0.5)) in nats, [B, K]."""
  p = p.clamp(1e-6, 1 - 1e-6)
  return p * (2 * p).log() + (1 - p) * (2 * (1 - p)).log()


class LatentAdapter(nn.Module):
  """Encoder + z projection, hooked into the backbone's sigma_map."""

  def __init__(self, backbone, bits, vocab_size, seq_len, cond_dim,
               enc_dim=256, enc_layers=2, enc_heads=4):
    super().__init__()
    self.bits, self.cond_dim = bits, cond_dim
    self.enc_cfg = dict(enc_dim=enc_dim, enc_layers=enc_layers,
                        enc_heads=enc_heads)
    if bits:
      self.encoder = SeqEncoder(vocab_size, seq_len, bits, enc_dim, enc_layers,
                                enc_heads, vocab_embedding(backbone, vocab_size))
      self.z_proj = nn.Linear(bits, cond_dim)
      nn.init.zeros_(self.z_proj.weight)
      nn.init.zeros_(self.z_proj.bias)
    self._z_emb = None
    self.hook_name, sigma_map = _find(backbone, 'sigma_map')
    self._handle = sigma_map.register_forward_hook(self._hook)

  def _hook(self, module, inputs, output):
    if self._z_emb is None:
      return output
    assert output.shape[0] == self._z_emb.shape[0], (
      f'latent batch {self._z_emb.shape[0]} != model batch {output.shape[0]}')
    return output + self._z_emb.to(output.dtype)

  def set_z(self, z):
    """Condition subsequent denoiser calls on bits z [B, K] (None clears)."""
    self._z_emb = None if (z is None or not self.bits) else self.z_proj(2 * z - 1)

  def sample_prior(self, n, device, generator=None):
    return torch.bernoulli(torch.full((n, self.bits), 0.5, device=device),
                           generator=generator)

  def remove(self):
    self._handle.remove()


def save_checkpoint(path, adapter, backbone, extra):
  torch.save({
    'bits': adapter.bits, 'cond_dim': adapter.cond_dim,
    'enc_cfg': adapter.enc_cfg,
    'adapter': {k: v for k, v in adapter.state_dict().items()
                if not k.endswith('encoder.emb')},   # frozen copy, rebuilt
    'adaln': {n: p.detach().cpu() for n, p in adaln_parameters(backbone)},
    **extra}, path)


def load_into(model, path, seq_len):
  """Attach a trained adapter (or a bits=0 control) to a Diffusion model.
  Copies the fine-tuned adaLN weights into the backbone. Returns the adapter."""
  ck = torch.load(path, map_location='cpu')
  params = dict(adaln_parameters(model.backbone))
  with torch.no_grad():
    for n, v in ck['adaln'].items():
      params[n].copy_(v.to(params[n].device, params[n].dtype))
  adapter = LatentAdapter(model.backbone, ck['bits'], model.vocab_size,
                          seq_len, ck['cond_dim'], **ck['enc_cfg'])
  missing, unexpected = adapter.load_state_dict(ck['adapter'], strict=False)
  assert not unexpected and all(k.endswith('encoder.emb') for k in missing), (
    missing, unexpected)
  adapter.to(model.device).eval()
  for p in adapter.parameters():
    p.requires_grad_(False)
  return adapter
