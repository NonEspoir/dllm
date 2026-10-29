"""FastAncestralSampler: MDLM ancestral sampling that only does float64
work at the positions revealed in the current step (D07).

The upstream AncestralSampler (samplers.py) samples a token for every
position in float64 over the whole vocabulary, then keeps only the masked
positions whose denoising coin came up. Given x_t, positions are
independent, so drawing the coins first and sampling tokens only where
they are needed gives exactly the same joint distribution.

Options:
  logits_mode
    full      call model.forward as upstream (same log-probs bit for bit),
              then gather the selected rows before the float64 cast.
    selected  run the backbone up to the final hidden states and apply the
              output layer only on the selected rows. Same ops, smaller
              matmul, so log-probs match up to bf16 rounding.
  noise_mode
    reference consume the RNG exactly like upstream (full-size Gumbel
              uniforms, then coins). Only for the token-exact test.
    fast      coins first, Gumbel uniforms only for the selected rows.
  skip_unchanged (selected mode only)
    MDLM here has no time conditioning, so a row whose tokens did not
    change since its last forward has the same hidden states. Cache them
    and only rerun the backbone for rows that changed and need logits.

The state also records, for every generated position, the step at which
it was revealed and the log-prob of the token written there (under the
temperature-scaled distribution actually sampled from). The RL rollout
uses these.
"""
import torch
import torch.nn.functional as F
from dataclass_patch import dataclass

import samplers
from models.dit import modulate_fused


@dataclass(kw_only=True)
class FastAncestralState(samplers.BaseState):
  xt: torch.Tensor
  timesteps: torch.Tensor
  ones: torch.Tensor
  start_idx: int
  step_idx: int
  nfe: int
  done: bool
  reveal_step: torch.Tensor            # (B, L) int64, -1 if never revealed
  token_logp: torch.Tensor             # (B, L) float64, 0 if never revealed
  hidden: torch.Tensor = None          # (B, L, D) cache for skip_unchanged
  t_cond: torch.Tensor = None          # (B, cond_dim) or None
  dirty: torch.Tensor = None           # (B,) bool, row changed since cached
  row_forwards: int = 0                # rows pushed through the backbone


def _backbone_hidden(backbone, xt, sigma):
  """DIT.forward up to (excluding) the final linear layer.

  Mirrors models/dit.py DIT.forward and DDiTFinalLayer.forward (no kv
  cache, non-causal). Returns the modulated hidden states and t_cond.
  """
  x = backbone.vocab_embed(xt)
  if backbone.adaLN:
    t_cond = F.silu(backbone.sigma_map(sigma))
  else:
    t_cond = None
  rotary_cos_sin = backbone.rotary_emb(x)
  with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    for block in backbone.blocks:
      x = block(x, rotary_cos_sin, c=t_cond, kv_cache=False)
    final = backbone.output_layer
    x = final.norm_final(x)
    if final.adaLN:
      shift, scale = final.adaLN_modulation(t_cond)[:, None].chunk(2, dim=2)
      x = modulate_fused(x, shift, scale)
  return x, t_cond


def _output_logits(backbone, h_rows):
  with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    return backbone.output_layer.linear(h_rows)


class FastAncestralSampler(samplers.Sampler):
  """Ancestral sampler for MDLM (absorbing diffusion only)."""

  def __init__(self, use_float64=True, noise_removal='ancestral',
               steps_policy='full', temperature=1.0,
               logits_mode='selected', noise_mode='fast',
               skip_unchanged=False):
    assert noise_removal in ('ancestral', 'greedy')
    assert steps_policy in ('full', 'proportional')
    assert logits_mode in ('full', 'selected')
    assert noise_mode in ('reference', 'fast')
    assert not (skip_unchanged and logits_mode != 'selected')
    self.use_float64 = use_float64
    self.noise_removal = noise_removal
    self.steps_policy = steps_policy
    self.temperature = temperature
    self.logits_mode = logits_mode
    self.noise_mode = noise_mode
    self.skip_unchanged = skip_unchanged

  def init_state(self, model, num_samples, *,
                 num_steps=None, eps=1e-5, prefix_tokens=None,
                 prefix_lengths=None):
    assert model.diffusion_type == 'absorbing', \
      'FastAncestralSampler only supports absorbing diffusion (MDLM)'
    if self.skip_unchanged:
      assert not model.time_conditioning, \
        'skip_unchanged requires a time-independent denoiser'
    # Same initialisation as samplers.AncestralSampler.init_state.
    self._validate_prefix_args(prefix_tokens, prefix_lengths)
    xt = model.prior_sample(num_samples, model.num_tokens)
    if prefix_tokens is not None:
      start_idx = int(prefix_lengths.min())
      self._project_prefix(xt, prefix_tokens, prefix_lengths)
    else:
      start_idx = 0
    if num_steps is None:
      num_steps = model.config.sampler.steps
    if self.steps_policy == 'proportional':
      gen_len = model.num_tokens - start_idx
      num_steps = int(round(num_steps * gen_len))
    timesteps = torch.linspace(1, eps, num_steps + 1,
                               device=model.device)
    ones = torch.ones(xt.shape[0], 1, dtype=model.dtype,
                      device=model.device)
    return FastAncestralState(
      xt=xt, timesteps=timesteps, ones=ones, start_idx=start_idx,
      step_idx=0, nfe=0, done=False,
      prefix_tokens=prefix_tokens, prefix_lengths=prefix_lengths,
      reveal_step=torch.full(xt.shape, -1, dtype=torch.long,
                             device=xt.device),
      token_logp=torch.zeros(xt.shape, dtype=torch.float64,
                             device=xt.device),
      dirty=torch.ones(xt.shape[0], dtype=torch.bool, device=xt.device))

  # ---- log-probs at the selected positions ---------------------------
  def _selected_log_probs_full(self, model, state, sigma_t, sel):
    log_probs = model.forward(
      xt=state.xt, sigma=sigma_t,
      context=samplers.AncestralContext(temperature=self.temperature))
    state.row_forwards += state.xt.shape[0]
    return log_probs[:, state.start_idx:][sel]

  def _selected_log_probs_selected(self, model, state, sigma_t, sel):
    backbone = model.backbone
    need = sel.any(dim=1)
    if self.skip_unchanged:
      # dirty starts all-True, so the first call runs exactly `need`.
      run = need & state.dirty
      if run.any():
        sigma = model._process_sigma(sigma_t[run])
        with torch.amp.autocast('cuda', dtype=torch.float32):
          h, t_cond = _backbone_hidden(backbone, state.xt[run], sigma)
        if state.hidden is None:
          state.hidden = h.new_zeros(state.xt.shape + (h.shape[-1],))
        state.hidden[run] = h
        state.dirty[run] = False
        state.row_forwards += int(run.sum())
      hidden = state.hidden
    else:
      sigma = model._process_sigma(sigma_t)
      with torch.amp.autocast('cuda', dtype=torch.float32):
        hidden, _ = _backbone_hidden(backbone, state.xt, sigma)
      state.row_forwards += state.xt.shape[0]
    h_rows = hidden[:, state.start_idx:][sel]
    with torch.amp.autocast('cuda', dtype=torch.float32):
      logits = _output_logits(backbone, h_rows)
    # Same post-processing as TrainerBase.forward + MDLM log_probs mode:
    # temperature, mask token suppression, log-softmax. All rows are
    # masked positions, so the "unmasked -> one-hot" branch is a no-op.
    if self.temperature != 1.0:
      logits = logits / self.temperature
    xt_rows = torch.full((logits.shape[0], 1), model.mask_index,
                         dtype=torch.long, device=logits.device)
    return model._process_model_output(
      model_output=logits[:, None, :], xt=xt_rows, sigma=None)[:, 0]

  # ---- one step -------------------------------------------------------
  def step(self, model, state):
    num_steps = len(state.timesteps) - 1
    is_last_step = (state.step_idx == num_steps - 1)
    t = state.timesteps[state.step_idx] * state.ones
    s = state.timesteps[state.step_idx + 1] * state.ones
    _, alpha_t = model.noise(t)
    if is_last_step:
      alpha_s = torch.ones_like(alpha_t)
    else:
      _, alpha_s = model.noise(s)
    sigma_t = model._sigma_from_alphat(alpha_t)

    window = state.xt[:, state.start_idx:]
    B, Lw = window.shape
    V = model.vocab_size
    noise_dtype = torch.float64 if self.use_float64 else torch.float32
    is_masked = (window == model.mask_index)
    removal = is_last_step  # last step reveals every remaining mask
    greedy = is_last_step and self.noise_removal == 'greedy'

    # 1. Coins. In reference mode, replay upstream's RNG order: the
    #    Gumbel uniforms for every (position, token) first, then coins.
    gumbel_u_full = None
    if self.noise_mode == 'reference' and not greedy:
      gumbel_u_full = torch.rand((B, Lw, V), dtype=noise_dtype,
                                 device=window.device)
    if removal:
      should_denoise = torch.ones_like(is_masked)
    else:
      denoise_prob = (alpha_s - alpha_t) / (1 - alpha_t)
      should_denoise = torch.rand(
        (B, Lw), dtype=torch.float64, device=window.device) < denoise_prob
    sel = is_masked & should_denoise

    # 2. Log-probs only where a token will be written.
    n_sel = int(sel.sum())
    if n_sel > 0:
      if self.logits_mode == 'full':
        log_probs = self._selected_log_probs_full(
          model, state, sigma_t, sel)
      else:
        log_probs = self._selected_log_probs_selected(
          model, state, sigma_t, sel)
      state.nfe += 1
      log_probs = samplers._maybe_cast_log_probs(log_probs, self.use_float64)

      # 3. Sample, same arithmetic as samplers.sample_categorical.
      if greedy:
        tokens = log_probs.argmax(dim=-1)
      else:
        probs = log_probs.exp()
        if gumbel_u_full is not None:
          u = gumbel_u_full[sel]
        else:
          u = torch.rand(probs.shape, dtype=probs.dtype,
                         device=probs.device)
        gumbel_noise = 1e-10 - (u + 1e-10).log()
        tokens = (probs / gumbel_noise).argmax(dim=-1)

      window[sel] = tokens
      b_idx, l_idx = sel.nonzero(as_tuple=True)
      l_idx = l_idx + state.start_idx
      state.reveal_step[b_idx, l_idx] = state.step_idx
      state.token_logp[b_idx, l_idx] = log_probs.gather(
        -1, tokens[:, None])[:, 0].to(torch.float64)
      state.dirty |= sel.any(dim=1)
    else:
      # Upstream still runs a forward here; we skip it.
      pass

    self._project_prefix(state.xt, state.prefix_tokens,
                         state.prefix_lengths)
    state.step_idx += 1
    if is_last_step:
      state.done = True
    return state

  def metadata(self, state):
    return {'nfe': state.nfe, 'row_forwards': state.row_forwards,
            'reveal_step': state.reveal_step,
            'token_logp': state.token_logp}
