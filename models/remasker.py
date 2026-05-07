"""Learned error-predictor head (g_phi) for guided star-shaped masked diffusion.

The remasker is a thin transformer head attached to a frozen DIT backbone.
It outputs a single logit per position: higher logit means "this token is
likely wrong and should be remasked". See Sec. 3 of
Meshchaninov et al., "Guided Star-Shaped Masked Diffusion" (arXiv:2510.08369).
"""

import omegaconf
import torch
import torch.nn as nn

from models.dit import DIT, LayerNorm, modulate_fused


class SingleLogitFinalLayer(nn.Module):
  """Replaces the DIT token-classification head with a single scalar per position."""

  def __init__(self, hidden_size: int, cond_dim: int):
    super().__init__()
    self.norm_final = LayerNorm(hidden_size)
    self.linear = nn.Linear(hidden_size, 1)

    # Zero-init for a stable start.
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()

    # AdaLN modulation from the sigma (timestep) embedding.
    self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()

  def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
    x_mod = modulate_fused(self.norm_final(x), shift, scale)
    return self.linear(x_mod)


class RemaskerNet(nn.Module):
  """Error-predictor g_phi implemented as a DIT backbone with a 1-logit head.

  Inputs:
    x_tokens: LongTensor of shape (batch, seq_len).

  Outputs:
    logits: FloatTensor of shape (batch, seq_len). Higher = more likely to remask.
  """

  def __init__(self, vocab_size: int, config: omegaconf.DictConfig):
    super().__init__()
    self.vocab_size = vocab_size
    self.seq_len = config.model.length

    # Optionally build a shallower DIT with only the first N transformer blocks.
    n_first = config.remasker.take_first_n_layers
    if n_first is not None:
      cfg_dict = omegaconf.OmegaConf.to_container(config, resolve=True)
      cfg_dict['model']['n_blocks'] = int(n_first)
      dit_config = cfg_dict  # DIT wraps dict into OmegaConf internally.
    else:
      dit_config = config

    self.dit = DIT(dit_config, vocab_size=vocab_size)

    # Replace the final projection with a single-logit head.
    self.output_layer = SingleLogitFinalLayer(
      hidden_size=config.model.hidden_size,
      cond_dim=config.model.cond_dim,
    )

  def change_final_layer(self):
    """Swap the DIT's token head for our single-logit head.

    Called right after construction (or after loading a checkpoint) so that
    `self.dit(...)` returns per-token error logits instead of vocab logits.
    """
    self.dit.output_layer = self.output_layer

  def freeze_backbone(self):
    for param in self.dit.parameters():
      param.requires_grad = False
    for param in self.output_layer.parameters():
      param.requires_grad = True

  def forward(self, x_tokens: torch.LongTensor) -> torch.FloatTensor:
    batch_size = x_tokens.shape[0]
    sigma = torch.zeros(batch_size, device=x_tokens.device, dtype=torch.float32)
    # DIT returns (batch, seq_len, 1); squeeze to (batch, seq_len).
    logits = self.dit(x_tokens, sigma)
    return logits.squeeze(-1)
