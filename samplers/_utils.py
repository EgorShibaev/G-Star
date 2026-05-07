"""Shared helpers for the sampler update functions."""

import torch


def sample_categorical(categorical_probs: torch.Tensor) -> torch.LongTensor:
  """Gumbel-argmax sample from a discrete distribution along the last dim."""
  gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
  return (categorical_probs / gumbel_norm).argmax(dim=-1)


def apply_nucleus_filter(p: torch.Tensor, nucleus_p: float) -> torch.Tensor:
  """Renormalize `p` so that only the top-`nucleus_p` cumulative mass remains."""
  sorted_probs, sorted_indices = torch.sort(p, descending=True, dim=-1)
  cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
  top_p_mask = cumulative_probs <= nucleus_p
  top_p_mask[..., 0] = True
  nucleus_probs = sorted_probs * top_p_mask
  nucleus_probs = nucleus_probs / nucleus_probs.sum(dim=-1, keepdim=True)
  return torch.zeros_like(p).scatter_(-1, sorted_indices, nucleus_probs)


def gumbel_top_k_positions(scaled_logits: torch.Tensor, counts: torch.LongTensor) -> torch.BoolTensor:
  """Select `counts[b]` positions per batch row via Gumbel-top-k sampling.

  Returns a bool mask of shape `scaled_logits.shape` with True at selected positions.
  """
  gumbel = -torch.log(-torch.log(torch.rand_like(scaled_logits) + 1e-10) + 1e-10)
  scores = scaled_logits + gumbel
  topk_indices = torch.argsort(scores, dim=1, descending=True)
  mask = torch.zeros_like(scaled_logits, dtype=torch.bool)
  batch_size = scaled_logits.shape[0]
  for b in range(batch_size):
    k = int(counts[b].item())
    if k > 0:
      mask[b, topk_indices[b, :k]] = True
  return mask
