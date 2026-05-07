"""Guided star-shaped sampler update using a learned error predictor g_phi.

Implements the per-step update from eq. (5) of
Meshchaninov et al., "Guided Star-Shaped Masked Diffusion" (arXiv:2510.08369).
"""

from typing import Tuple

import torch

from ._utils import apply_nucleus_filter, gumbel_top_k_positions, sample_categorical


def remasker_update(model, x: torch.LongTensor, t: torch.Tensor, dt: torch.Tensor) -> Tuple[torch.LongTensor, torch.LongTensor]:
  """One step of the guided star-shaped sampler.

  Pipeline:
    1. compute p_x0 with the denoiser at time t
    2. sample a fully-denoised proposal x_0_hat ~ p_x0 at masked positions
       (keeping already-revealed tokens unchanged)
    3. run the remasker on x_prop to obtain per-token error logits
    4. temperature-scale and perturb the logits with Gumbel noise
    5. select N = ceil(alpha_s * L) positions to remask, where s = t - dt

  Returns (x_next, x_0_hat).
  """
  sigma_t, _ = model.noise(t)
  if sigma_t.ndim > 1:
    sigma_t = sigma_t.squeeze(-1)
  log_p_x0 = model.forward(x, sigma_t)

  # Temperature-scale logits during remasking proposal sampling.
  denoiser_temp = max(1e-6, float(model.config.sampling.denoiser_temp_during_remasking))
  scaled_log_p_x0 = log_p_x0 / denoiser_temp
  # Subtract max-logit for numerical stability, then softmax.
  scaled_log_p_x0 = scaled_log_p_x0 - scaled_log_p_x0.max(dim=-1, keepdim=True).values
  p_x0 = scaled_log_p_x0.exp()
  p_x0 = p_x0 / p_x0.sum(dim=-1, keepdim=True)

  if model.config.sampling.nucleus_p < 1:
    p_x0 = apply_nucleus_filter(p_x0, float(model.config.sampling.nucleus_p))

  if model.config.sampling.use_fp64:
    p_x0 = p_x0.to(torch.float64)

  _x = sample_categorical(p_x0)
  copy_flag = (x != model.mask_index).to(x.dtype)
  x_prop = copy_flag * x + (1 - copy_flag) * _x

  if model.remasker is not None:
    with torch.no_grad():
      pos_logits = model.remasker(x_prop)  # (batch, seq_len)
    temperature = max(1e-6, float(model.remasker_temperature))
    scaled = pos_logits / temperature
  else:
    # Baseline ablation: no remasker loaded -> random selection.
    batch, seq_len = x_prop.shape
    scaled = torch.randn(batch, seq_len, device=x_prop.device)
    pos_logits = scaled

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

  if model.config.sampling.remasker_adaptive_sampling:
    # Adaptive variant: remask every position whose error logit is above -1.
    pos_logits_copy = pos_logits.clone()
    pos_logits_copy[padding_mask] = -torch.inf
    cnt_pos_logits = (pos_logits_copy > -1).sum(dim=-1)
    num_remask = torch.clamp(cnt_pos_logits, min=num_remask * 0.0, max=num_remask * 10000.0)

  remask_positions = gumbel_top_k_positions(scaled, num_remask)

  x_next = x_prop.clone()
  x_next[remask_positions] = model.mask_index
  return x_next, _x
