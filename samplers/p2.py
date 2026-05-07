"""Path-Planning (P2) sampler baseline (Peng et al., 2025).

At every step, P2 resamples a proposal for every position (revision is allowed
on already-unmasked tokens) and keeps the top-(1 - alpha_s) * L most confident
positions.
"""

from typing import Tuple

import torch

from ._utils import apply_nucleus_filter, sample_categorical


def p2_update(model, x: torch.LongTensor, t: torch.Tensor, dt: torch.Tensor) -> Tuple[torch.LongTensor, torch.LongTensor]:
  """One P2 update step.

  Returns (x_next, y), where `y` is the proposal sequence (for trajectory logs).
  """
  sigma_t, _ = model.noise(t)
  if sigma_t.ndim > 1:
    sigma_t = sigma_t.squeeze(-1)
  log_p = model.forward(x, sigma_t)

  temp = max(1e-6, float(model.config.p2.temperature))
  log_p = log_p / temp
  log_p = log_p - log_p.max(dim=-1, keepdim=True).values
  p = log_p.exp()
  p = p / p.sum(dim=-1, keepdim=True)

  if model.config.p2.nucleus_p < 1:
    p = apply_nucleus_filter(p, float(model.config.p2.nucleus_p))

  if model.config.sampling.use_fp64:
    p = p.to(torch.float64)

  # Sample a fresh proposal for every position so that already-unmasked tokens
  # can also be revised by the top-k selection below.
  if model.config.p2.use_argmax:
    y = p.argmax(dim=-1)
  else:
    y = sample_categorical(p)

  if model.config.p2.confidence_type == 'max_prob':
    conf = p.max(dim=-1).values
  else:
    conf = p.gather(-1, y.unsqueeze(-1)).squeeze(-1)

  padding_mask = (x == model.tokenizer.pad_token_id)
  conf[padding_mask] = -float('inf')

  sigma_s, _ = model.noise(t - dt)
  if sigma_s.ndim > 1:
    sigma_s = sigma_s.squeeze(-1)
  alpha_s = 1 - torch.exp(-sigma_s)

  seq_lens = (~padding_mask).sum(dim=-1)
  num_keep = ((1 - alpha_s) * seq_lens.float()).ceil().long()
  num_keep = torch.clamp(num_keep, min=0)
  num_keep = torch.minimum(num_keep, seq_lens)

  batch_size, _ = x.shape
  topk_indices = torch.argsort(conf, dim=1, descending=True)

  keep_token = torch.zeros_like(x, dtype=torch.bool)
  for b in range(batch_size):
    k = int(num_keep[b].item())
    if k > 0:
      keep_token[b, topk_indices[b, :k]] = True

  x_next = torch.full_like(x, model.mask_index)
  x_next[keep_token] = y[keep_token]
  x_next[padding_mask] = model.tokenizer.pad_token_id
  return x_next, y
