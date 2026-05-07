"""Confidence-guided star-shaped sampler (no learned remasker).

Uses the denoiser's own per-token confidence (from `p_x0` at the position that
was selected last step) as the error score. Designed for the `loop` noise
schedule.
"""

from typing import Tuple

import torch

from ._utils import apply_nucleus_filter, gumbel_top_k_positions, sample_categorical


def conf_star_shape_update(model, x: torch.LongTensor, t: torch.Tensor, dt: torch.Tensor, conf: torch.Tensor) -> Tuple[torch.LongTensor, torch.Tensor, torch.LongTensor]:
  """One step of the confidence-guided star-shaped sampler.

  Returns (x_next, conf_next, x_0_hat).
  """
  assert model.config.noise.type == "loop"
  sigma_t, _ = model.noise(t)
  if sigma_t.ndim > 1:
    sigma_t = sigma_t.squeeze(-1)
  log_p_x0 = model.forward(x, sigma_t)
  if model.config.sampling.nucleus_p < 1:
    p_x0 = apply_nucleus_filter(log_p_x0.exp(), float(model.config.sampling.nucleus_p))
  else:
    p_x0 = log_p_x0.exp()

  if model.config.sampling.use_fp64:
    p_x0 = p_x0.to(torch.float64)

  _x = sample_categorical(p_x0)
  copy_flag = (x != model.mask_index).to(x.dtype)
  x_prop = copy_flag * x + (1 - copy_flag) * _x

  # Build per-position "wrongness" logits:
  #   - for already-unmasked tokens: reuse cached confidence.
  #   - for newly proposed tokens:   use -p(token).
  was_masked = (x == model.mask_index)
  pos_logits = conf.to(dtype=p_x0.dtype).clone()
  proposed_token_probs = p_x0.gather(-1, _x.unsqueeze(-1)).squeeze(-1)
  pos_logits[was_masked] = -proposed_token_probs[was_masked]

  temperature = max(1e-6, float(model.config.sampling.remasker_temperature))
  scaled = pos_logits / temperature
  padding_mask = x_prop == model.tokenizer.pad_token_id
  scaled[padding_mask] = -torch.inf

  sigma_s, _ = model.noise(t - dt)
  if sigma_s.ndim > 1:
    sigma_s = sigma_s.squeeze(-1)
  alpha_s = 1 - torch.exp(-sigma_s)
  seq_lens = (x_prop != model.tokenizer.pad_token_id).sum(dim=-1)
  num_remask = torch.ceil(alpha_s * seq_lens).to(torch.int64)
  num_remask[num_remask > seq_lens] = seq_lens[num_remask > seq_lens]
  num_remask[num_remask < 0] = 0

  remask_positions = gumbel_top_k_positions(scaled, num_remask)

  x_next = x_prop.clone()
  x_next[remask_positions] = model.mask_index

  # Update cached confidences:
  #   - newly revealed tokens get -p(token).
  #   - remasked tokens are set to -inf.
  conf_next = conf.clone()
  unmask_mask = (x == model.mask_index) & (x_next != model.mask_index)
  kept_token_probs = p_x0.gather(-1, x_prop.unsqueeze(-1)).squeeze(-1)
  conf_next[unmask_mask] = (-kept_token_probs)[unmask_mask]
  conf_next[remask_positions] = -torch.inf

  return x_next, conf_next, _x
