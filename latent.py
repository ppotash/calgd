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
               emb_init=None, emb_dim=768, head_std=0.02):
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
    # Small random init: bits start near p = 0.5 (KL ~ 0) but already depend on
    # the text. A zero head plus the zero z projection would be a saddle where
    # neither receives a useful gradient.
    nn.init.normal_(self.head.weight, std=head_std)
    nn.init.zeros_(self.head.bias)

  def forward(self, x0):
    h = self.proj(self.emb[x0]) + self.pos[:x0.shape[1]]
    return self.head(self.norm(self.body(h)).mean(1))


class OracleEncoder(nn.Module):
  """Fixed, interpretable bits: bit k = 1 iff token tokens[k] occurs anywhere in
  the sequence. Returns saturated logits so z is deterministic. Used to test
  whether the denoiser can use document-level information at all."""

  def __init__(self, bits):
    super().__init__()
    self.register_buffer('tokens', torch.zeros(bits, dtype=torch.long))

  def presence(self, x):
    return (x.unsqueeze(-1) == self.tokens).any(1)                  # [B, K]

  def forward(self, x0):
    return self.presence(x0).float() * 40.0 - 20.0


def select_oracle_tokens(chunks, tokenizer, k, vocab_size):
  """k word tokens whose document frequency over `chunks` [N, L] is closest to
  0.5 (most informative bits). Prefers plain words (' said', ' game', ...).
  Returns (token ids, their document frequencies)."""
  import re
  df = torch.zeros(vocab_size)
  for row in chunks:
    df[torch.unique(row)] += 1
  df /= len(chunks)
  order = torch.argsort((df - 0.5).abs()).tolist()

  def is_word(v):
    try:
      return re.fullmatch(r' [A-Za-z]{3,}', tokenizer.decode([v])) is not None
    except Exception:
      return True
  chosen = [v for v in order if is_word(v)][:k]
  if len(chosen) < k:   # fall back to any tokens if too few plain words
    chosen += [v for v in order if v not in chosen][:k - len(chosen)]
  return torch.tensor(chosen), df[chosen]


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
               enc_dim=256, enc_layers=2, enc_heads=4, input_inject=False,
               head_std=0.02, oracle=None):
    super().__init__()
    self.bits, self.cond_dim = bits, cond_dim
    self.oracle = oracle if bits else None
    self.input_inject = bool(input_inject and bits)
    self.enc_cfg = dict(enc_dim=enc_dim, enc_layers=enc_layers,
                        enc_heads=enc_heads)
    emb = vocab_embedding(backbone, vocab_size)
    if bits and oracle == 'presence':
      self.encoder = OracleEncoder(bits)
    elif bits:
      assert oracle is None, f'unknown oracle {oracle!r}'
      self.encoder = SeqEncoder(vocab_size, seq_len, bits, enc_dim, enc_layers,
                                enc_heads, emb, head_std=head_std)
    if bits:
      self.z_proj = nn.Linear(bits, cond_dim)
      nn.init.zeros_(self.z_proj.weight)
      nn.init.zeros_(self.z_proj.bias)
    self._z_emb, self._z_in = None, None
    self.hook_name, sigma_map = _find(backbone, 'sigma_map')
    self._handles = [sigma_map.register_forward_hook(self._hook)]
    if self.input_inject:
      # z also added to every token embedding: a direct path into the trunk,
      # not only through each layer's scale/shift.
      assert emb is not None, 'input_inject needs the token embedding size'
      self.z_in_proj = nn.Linear(bits, emb.shape[1])
      nn.init.zeros_(self.z_in_proj.weight)
      nn.init.zeros_(self.z_in_proj.bias)
      name, vocab_embed = _find(backbone, 'vocab_embed')
      self.hook_name += f' + {name}'
      self._handles.append(vocab_embed.register_forward_hook(self._in_hook))

  def _hook(self, module, inputs, output):
    if self._z_emb is None:
      return output
    assert output.shape[0] == self._z_emb.shape[0], (
      f'latent batch {self._z_emb.shape[0]} != model batch {output.shape[0]}')
    return output + self._z_emb.to(output.dtype)

  def _in_hook(self, module, inputs, output):
    if self._z_in is None:
      return output
    return output + self._z_in.to(output.dtype).unsqueeze(1)

  def set_z(self, z):
    """Condition subsequent denoiser calls on bits z [B, K] (None clears)."""
    if z is None or not self.bits:
      self._z_emb, self._z_in = None, None
      return
    zs = 2 * z - 1
    self._z_emb = self.z_proj(zs)
    self._z_in = self.z_in_proj(zs) if self.input_inject else None

  def sample_prior(self, n, device, generator=None):
    return torch.bernoulli(torch.full((n, self.bits), 0.5, device=device),
                           generator=generator)

  def remove(self):
    for h in self._handles:
      h.remove()


def save_checkpoint(path, adapter, backbone, extra):
  torch.save({
    'bits': adapter.bits, 'cond_dim': adapter.cond_dim,
    'enc_cfg': adapter.enc_cfg, 'input_inject': adapter.input_inject,
    'oracle': adapter.oracle,
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
                          seq_len, ck['cond_dim'], **ck['enc_cfg'],
                          input_inject=ck.get('input_inject', False),
                          oracle=ck.get('oracle'))
  missing, unexpected = adapter.load_state_dict(ck['adapter'], strict=False)
  assert not unexpected and all(k.endswith('encoder.emb') for k in missing), (
    missing, unexpected)
  adapter.to(model.device).eval()
  for p in adapter.parameters():
    p.requires_grad_(False)
  return adapter
