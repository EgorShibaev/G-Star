"""Training entry point for the learned remasker (error predictor g_phi).

The remasker is trained against a frozen denoiser. For each x0 from the dataset
we sample t, build x_t by forward masking, run a single DDPM denoising step to
produce x_s, and then train g_phi to predict whether each token in x_s is an
error.

Supported label-generation strategies:
  - default          : compare 1-step xs to ground-truth x0 (mismatch = error).
  - random_corruption: corrupt a fraction of x0 tokens with random ones and use
                       the corruption mask as the target. No denoiser pipeline.
  - ar_perplexity    : run the default denoiser pipeline and label a position as
                       an error iff per-token NLL under a frozen AR model
                       exceeds `remasker.ar_nll_threshold`.

Supported losses:
  - BCE (default; optional class-balanced reweighting).
  - RankNet pairwise: within-sequence ranking loss that pushes error positions
    above correct ones.

See Meshchaninov et al., "Guided Star-Shaped Masked Diffusion" (arXiv:2510.08369),
Section 3 and Algorithm 1.
"""

import os
from typing import Tuple

import hydra
import lightning as L
import omegaconf
import torch
import torch.nn.functional as F
import transformers
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

import dataloader
import diffusion as diffusion_mod
from main import _print_config
from models.remasker import RemaskerNet


def compute_ranknet_pairwise_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
  """Compute RankNet pairwise ranking loss within each sequence.

  Encourages positive samples (error tokens, label=1) to have higher scores
  than negatives (correct tokens, label=0), so that higher-logit positions are
  preferred for remasking. Computed independently per sequence and averaged
  across sequences with at least one positive and one negative position:

    L = (1 / BS) * sum_b [ (1 / (|P_b| |H_b|)) * sum_{i in P_b, j in H_b}
                            log(1 + exp(s_j - s_i)) ]

  Args:
    logits: Model output logits, shape (BS, N).
    labels: Binary labels, shape (BS, N) (1 = should remask, 0 = correct).
    loss_mask: Positions to include in the loss, shape (BS, N).
    device: Device for tensor operations.
  """
  batch_size = logits.shape[0]
  total_loss = torch.tensor(0.0, device=device, dtype=logits.dtype)
  valid_seqs = 0

  for b in range(batch_size):
    seq_mask = loss_mask[b]
    seq_logits = logits[b][seq_mask]
    seq_labels = labels[b][seq_mask]

    if seq_logits.numel() == 0:
      continue

    pos_mask = seq_labels > 0.5
    neg_mask = seq_labels <= 0.5

    pos_logits = seq_logits[pos_mask]
    neg_logits = seq_logits[neg_mask]

    num_pos = pos_logits.numel()
    num_neg = neg_logits.numel()

    if num_pos == 0 or num_neg == 0:
      continue

    # diff[i, j] = neg_logits[j] - pos_logits[i]. We want pos > neg, so
    # loss = softplus(neg - pos).
    diff = neg_logits.unsqueeze(0) - pos_logits.unsqueeze(1)
    pairwise_loss = F.softplus(diff)

    seq_loss = pairwise_loss.sum() / (num_pos * num_neg)
    total_loss = total_loss + seq_loss
    valid_seqs += 1

  if valid_seqs > 0:
    return total_loss / valid_seqs
  return torch.zeros((), device=device, dtype=logits.dtype, requires_grad=True)


class RemaskerModule(L.LightningModule):
  """Lightning module that trains a RemaskerNet against a frozen denoiser.

  Default training step:
    - Sample x0 from a batch.
    - Sample t ~ Uniform(eps, 1), s = 0, dt = t - s.
    - Compute x_t via q(x_t | x0, t).
    - Compute x_s via a single DDPM update from t to s.
    - Label: target = 1 where x_s != x0, else 0.
    - Optimize BCE (or RankNet) on those positions (optionally only newly
      generated tokens).
  """

  def __init__(self, config: omegaconf.DictConfig, tokenizer, denoiser: diffusion_mod.Diffusion):
    super().__init__()
    self.save_hyperparameters(ignore=['tokenizer', 'denoiser'])
    self.config = config
    self.tokenizer = tokenizer
    self.denoiser = denoiser.eval()
    for p in self.denoiser.parameters():
      p.requires_grad = False

    # Common attributes borrowed from the denoiser.
    self.mask_index = int(self.denoiser.mask_index)
    self.vocab_size = int(self.denoiser.vocab_size)
    self.padding_index = int(self.denoiser.tokenizer.pad_token_id)
    self.seq_len = int(self.config.model.length)
    self.eps = float(self.config.training.sampling_eps)

    # t-sampling mode for training: 'uniform' or 'const'.
    self.t_sampling = config.sampling.t_sampling
    self.t_const = config.sampling.t_const

    # Training strategy: 'default', 'random_corruption', or 'ar_perplexity'.
    self.training_strategy = str(getattr(config.remasker, 'training_strategy', 'default'))
    self.corruption_ratio = float(getattr(config.remasker, 'corruption_ratio', 0.1))
    self.ar_nll_threshold = float(getattr(config.remasker, 'ar_nll_threshold', 3.0))

    # Lazily-loaded AR model for the ar_perplexity strategy.
    self._ar_model = None
    self._ar_model_name = str(getattr(config.remasker, 'ar_model_name', 'gpt2'))

    self.net = RemaskerNet(vocab_size=self.vocab_size, config=self.config)

    self.lr = float(self.config.optim.lr)

  @torch.no_grad()
  def _sample_t_s(self, batch_size: int, device: torch.device) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Sample t (and keep s = 0, dt = t) according to the configured mode."""
    eps = torch.tensor(self.eps, device=device, dtype=torch.float32)

    if self.t_sampling == 'uniform':
      t = (1 - eps) * torch.rand(batch_size, device=device) + eps
    elif self.t_sampling == 'const':
      t_value = max(float(self.eps), min(1.0, float(self.t_const)))
      t = torch.full((batch_size,), t_value, device=device, dtype=torch.float32)
    else:
      raise ValueError(f"Unknown t_sampling mode: {self.t_sampling}. Expected 'uniform' or 'const'.")

    s = 0
    dt = t - s
    return t, s, dt

  @torch.no_grad()
  def _compute_xt_xs_and_labels(self, x0: torch.LongTensor) -> Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor, torch.FloatTensor]:
    device = x0.device
    batch_size = x0.shape[0]
    padding_mask = x0 == self.padding_index

    # Sample times.
    t, s, dt = self._sample_t_s(batch_size, device)
    t_in = t.view(batch_size, 1)
    dt_in = dt.view(batch_size, 1)

    # Compute q(x_t | x0).
    sigma_t, _ = self.denoiser.noise(t)
    move_chance = 1.0 - torch.exp(-sigma_t)
    if move_chance.ndim == 1:
      move_chance = move_chance[:, None]
    xt = self.denoiser.q_xt(x0, move_chance)

    if not self.config.sampling.fix_train_inference_mismatch:
      # Single DDPM step from t to s; augmentations (below) are then applied to
      # non-padding positions of xs, covering both originally-masked and
      # originally-unmasked tokens.
      xs, _ = self.denoiser._ddpm_update(xt, t_in, dt_in)
    else:
      assert s == 0, "s should be 0 when fix_train_inference_mismatch is True"
      # Mirror the inference pipeline:
      #   1. sample x'_0 ~ p(x'_0 | x_t, t)
      #   2. refine via the current training remasker: x'_t ~ q(x'_t | x'_0)
      #   3. denoise x'_t down to s = 0.
      # Steps 1+2 are performed by _remasker_update (with dt = 0 so the time
      # does not change), step 3 by _ddpm_update.
      prev_remasker = self.denoiser.remasker
      self.denoiser.remasker = self.net
      self.denoiser.remasker_temperature = float(self.config.sampling.remasker_temperature)
      zero_dt = torch.zeros_like(dt_in)
      x_t_prime, _ = self.denoiser._remasker_update(xt, t_in, zero_dt)
      self.denoiser.remasker = prev_remasker
      xs, _ = self.denoiser._ddpm_update(x_t_prime, t_in, dt_in)

    # Apply training-time augmentations to all non-padding positions of xs.
    eligible_mask = (~padding_mask)
    num_eligible = int(eligible_mask.sum().item())

    if num_eligible > 0:
      valid_tokens = torch.arange(self.vocab_size, device=xt.device)
      valid_tokens = valid_tokens[(valid_tokens != self.mask_index) & (valid_tokens != self.padding_index)]

      uniform_ratio = float(self.config.sampling.uniform_ratio)
      prior_ratio = float(self.config.sampling.prior_ratio)
      semantic_ratio = float(getattr(self.config.sampling, 'semantic_change_ratio', 0.0))
      interval_ratio = float(getattr(self.config.sampling, 'interval_shuffle_ratio', 0.0))

      if semantic_ratio > 0.0:
        # Build a normalized vocabulary embedding matrix once for cosine similarity.
        with torch.no_grad():
          embed_layer = self.denoiser.backbone.vocab_embed
          base_emb = torch.cat([embed_layer.embeddings, embed_layer.new_embedding], dim=0)
          base_emb = base_emb.to(device=xt.device, dtype=torch.float32)
          base_emb = F.normalize(base_emb, p=2, dim=-1)

      for b in range(batch_size):
        pos_b = torch.nonzero(eligible_mask[b], as_tuple=False).squeeze(1)
        num_b = int(pos_b.numel())
        if num_b == 0:
          continue

        # Uniform-random replacement augmentation.
        if uniform_ratio > 0.0 and valid_tokens.numel() > 0:
          k_uniform_b = int(round(uniform_ratio * num_b))
          k_uniform_b = max(0, min(k_uniform_b, num_b))
          if k_uniform_b > 0:
            perm_b = torch.randperm(num_b, device=xt.device)[:k_uniform_b]
            sel_pos_b = pos_b[perm_b]
            sampled = valid_tokens[torch.randint(low=0, high=valid_tokens.numel(), size=(k_uniform_b,), device=xt.device)]
            xs[b, sel_pos_b] = sampled

        # Token-frequency prior replacement.
        if prior_ratio > 0.0:
          k_prior_b = int(round(prior_ratio * num_b))
          k_prior_b = max(0, min(k_prior_b, num_b))
          if k_prior_b > 0:
            perm_b = torch.randperm(num_b, device=xt.device)[:k_prior_b]
            sel_pos_b = pos_b[perm_b]
            sentence_tokens = x0[b][(x0[b] != self.padding_index) & (x0[b] != self.mask_index)]
            if sentence_tokens.numel() == 0 or sentence_tokens.dtype != torch.long:
              if valid_tokens.numel() > 0:
                sampled = valid_tokens[torch.randint(low=0, high=valid_tokens.numel(), size=(k_prior_b,), device=xt.device)]
                xs[b, sel_pos_b] = sampled
            else:
              unique_tokens, counts = torch.unique(sentence_tokens, return_counts=True)
              probs = counts.to(dtype=torch.float32)
              probs = probs / probs.sum()
              sampled_idx = torch.multinomial(probs, num_samples=k_prior_b, replacement=True)
              sampled = unique_tokens[sampled_idx]
              xs[b, sel_pos_b] = sampled

        # Semantic-change augmentation: sample replacements by cosine similarity.
        if semantic_ratio > 0.0 and num_b > 0:
          k_sem_b = int(round(semantic_ratio * num_b))
          k_sem_b = max(0, min(k_sem_b, num_b))
          if k_sem_b > 0:
            perm_b = torch.randperm(num_b, device=xt.device)[:k_sem_b]
            sel_pos_b = pos_b[perm_b]
            current_tokens = xs[b, sel_pos_b].to(torch.long)
            with torch.no_grad():
              token_emb = base_emb[current_tokens]
              sims = torch.matmul(base_emb, token_emb.T).to(torch.float32)
              sims[self.padding_index, :] = -float('inf')
              sims[self.mask_index, :] = -float('inf')
              col_indices = torch.arange(k_sem_b, device=xt.device)
              sims[current_tokens, col_indices] = -float('inf')
              sims = sims - sims.max(dim=0, keepdim=True).values
              probs = torch.softmax(sims, dim=0)
              sampled_ids = torch.multinomial(probs.T, num_samples=1, replacement=True).squeeze(1)
            xs[b, sel_pos_b] = sampled_ids.to(xs.dtype)

        # Interval-shuffle augmentation: shuffle tokens inside random length-4-8 intervals.
        if interval_ratio > 0.0 and num_b > 0:
          k_shuffle_b = int(round(interval_ratio * num_b))
          k_shuffle_b = max(0, min(k_shuffle_b, num_b))
          affected = 0
          while affected < k_shuffle_b and num_b >= 4:
            length = int(torch.randint(low=4, high=9, size=(1,), device=xt.device).item())
            length = min(length, k_shuffle_b - affected)
            if length < 4 or num_b - length <= 0:
              break
            start = int(torch.randint(low=0, high=num_b - length + 1, size=(1,), device=xt.device).item())
            idx_slice = pos_b[start: start + length]
            segment = xs[b, idx_slice]
            perm = torch.randperm(length, device=xt.device)
            xs[b, idx_slice] = segment[perm]
            affected += length

    # Optional: self-consistency target compares 1-step xs to k-step xk.
    use_self_consistency = self.config.training.remasker_self_consistency
    k_steps = int(self.config.training.remasker_self_consistency_k)
    if use_self_consistency and k_steps > 1:
      x_curr = xt
      t_curr = t_in.clone()
      step_dt = dt_in / float(k_steps)
      for _ in range(k_steps):
        x_next, _ = self.denoiser._ddpm_update(x_curr, t_curr, step_dt)
        x_curr = x_next
        t_curr = t_curr - step_dt
      xk = x_curr
    else:
      xk = xs

    # Newly-generated positions between xt and xs.
    new_mask = (xt == self.mask_index) & (xs != self.mask_index)

    if use_self_consistency and k_steps > 1:
      target = (xs != xk).to(torch.float32)
    else:
      target = (xs != x0).to(torch.float32)

    xt[padding_mask] = self.padding_index
    xs[padding_mask] = self.padding_index
    xk[padding_mask] = self.padding_index
    new_mask[padding_mask] = False
    target[padding_mask] = 0

    return xt, xs, new_mask.to(torch.bool), target

  @torch.no_grad()
  def _compute_random_corruption_labels(
    self, x0: torch.LongTensor
  ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor, torch.FloatTensor]:
    """Strategy: corrupt a fraction of tokens with random vocabulary tokens.

    No diffusion pipeline is used; the corruption mask is the target.
    Returns (None, x_corrupted, all-true mask, target).
    """
    device = x0.device
    batch_size, _ = x0.shape
    padding_mask = x0 == self.padding_index

    valid_tokens = torch.arange(self.vocab_size, device=device)
    valid_tokens = valid_tokens[
      (valid_tokens != self.mask_index) & (valid_tokens != self.padding_index)
    ]

    x_corrupted = x0.clone()
    target = torch.zeros_like(x0, dtype=torch.float32)

    for b in range(batch_size):
      eligible = (~padding_mask[b]).nonzero(as_tuple=False).squeeze(1)
      num_eligible = eligible.numel()
      if num_eligible == 0:
        continue
      k = max(1, int(round(self.corruption_ratio * num_eligible)))
      k = min(k, num_eligible)
      perm = torch.randperm(num_eligible, device=device)[:k]
      chosen_pos = eligible[perm]
      random_tokens = valid_tokens[
        torch.randint(0, valid_tokens.numel(), (k,), device=device)
      ]
      x_corrupted[b, chosen_pos] = random_tokens
      target[b, chosen_pos] = (random_tokens != x0[b, chosen_pos]).float()

    target[padding_mask] = 0
    new_mask = ~padding_mask
    return None, x_corrupted, new_mask, target

  @torch.no_grad()
  def _compute_ar_perplexity_labels(
    self, x0: torch.LongTensor
  ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor, torch.FloatTensor]:
    """Strategy: run the denoiser pipeline, then label tokens using AR per-token NLL."""
    device = x0.device
    batch_size = x0.shape[0]
    padding_mask = x0 == self.padding_index

    # Run the default denoiser pipeline to produce xs.
    t, _, dt = self._sample_t_s(batch_size, device)
    t_in = t.view(batch_size, 1)
    dt_in = dt.view(batch_size, 1)

    sigma_t, _ = self.denoiser.noise(t)
    move_chance = 1.0 - torch.exp(-sigma_t)
    if move_chance.ndim == 1:
      move_chance = move_chance[:, None]
    xt = self.denoiser.q_xt(x0, move_chance)

    xs, _ = self.denoiser._ddpm_update(xt, t_in, dt_in)

    # Prepare xs for the AR model by replacing stray mask/padding tokens with EOS.
    eos_id = self.tokenizer.eos_token_id or 0
    xs_for_ar = xs.clone()
    xs_for_ar[xs_for_ar == self.mask_index] = eos_id
    xs_for_ar[padding_mask] = eos_id

    if self._ar_model is None:
      self._ar_model = transformers.AutoModelForCausalLM.from_pretrained(
        self._ar_model_name
      ).eval().to(device)
      for p in self._ar_model.parameters():
        p.requires_grad = False

    ar_outputs = self._ar_model(xs_for_ar)
    ar_logits = ar_outputs.logits  # (batch, seq_len, vocab)

    # Per-token NLL: compare prediction at position i-1 to the token at position i.
    shift_logits = ar_logits[:, :-1, :].contiguous()
    shift_labels = xs_for_ar[:, 1:].contiguous()
    per_token_nll = F.cross_entropy(
      shift_logits.view(-1, shift_logits.size(-1)),
      shift_labels.view(-1),
      reduction='none',
    ).view(batch_size, -1)

    # First position has no predecessor, so pad with 0.
    per_token_nll = torch.cat(
      [torch.zeros(batch_size, 1, device=device), per_token_nll], dim=1
    )

    target = (per_token_nll > self.ar_nll_threshold).float()
    target[padding_mask] = 0

    new_mask = (xt == self.mask_index) & (xs != self.mask_index)
    new_mask[padding_mask] = False

    xt[padding_mask] = self.padding_index
    xs[padding_mask] = self.padding_index

    return xt, xs, new_mask, target

  def _compute_loss_and_metrics(self, logits: torch.FloatTensor, target: torch.FloatTensor, new_mask: torch.BoolTensor):
    use_ranknet = bool(getattr(self.config.training, 'remasker_use_ranknet_pairwise_loss', False))

    if self.config.training.remasker_loss_only_new_tokens:
      loss_mask_2d = new_mask.to(torch.bool)
    else:
      loss_mask_2d = torch.ones_like(target, dtype=torch.bool)

    if use_ranknet:
      loss = compute_ranknet_pairwise_loss(
        logits=logits,
        labels=target,
        loss_mask=loss_mask_2d,
        device=logits.device,
      )
      selected_logits = logits[loss_mask_2d]
      selected_target = target[loss_mask_2d]
    else:
      selected_logits = logits[loss_mask_2d]
      selected_target = target[loss_mask_2d]

      if selected_logits.numel() == 0:
        loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
      else:
        if self.config.training.remasker_reweighting:
          with torch.no_grad():
            total_elems = torch.tensor(float(selected_target.numel()), device=logits.device, dtype=logits.dtype)
            pos_count = selected_target.sum()
            neg_count = total_elems - pos_count
            pos_count = torch.clamp(pos_count, min=1.0)
            neg_count = torch.clamp(neg_count, min=1.0)
            w_pos = (total_elems / (2.0 * pos_count)).to(dtype=logits.dtype)
            w_neg = (total_elems / (2.0 * neg_count)).to(dtype=logits.dtype)
            weight = torch.where(selected_target > 0.5, w_pos, w_neg)
          loss = F.binary_cross_entropy_with_logits(selected_logits, selected_target, weight=weight, reduction='mean')
        else:
          loss = F.binary_cross_entropy_with_logits(selected_logits, selected_target, reduction='mean')

    with torch.no_grad():
      if selected_logits.numel() == 0:
        acc = torch.tensor(0.0, device=logits.device)
        precision = torch.tensor(0.0, device=logits.device)
        recall = torch.tensor(0.0, device=logits.device)
        f1 = torch.tensor(0.0, device=logits.device)
        mistake_ratio = torch.tensor(0.0, device=logits.device)
      else:
        preds = (torch.sigmoid(selected_logits) > 0.5).to(torch.int32)
        true = selected_target.to(torch.int32)

        total = preds.numel()
        correct = (preds == true).sum()
        acc = correct.float() / total

        tp = ((preds == 1) & (true == 1)).sum().float()
        fp = ((preds == 1) & (true == 0)).sum().float()
        fn = ((preds == 0) & (true == 1)).sum().float()

        precision = tp / (tp + fp).clamp(min=1e-8)
        recall = tp / (tp + fn).clamp(min=1e-8)
        f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)

        mistakes = (true == 1).sum().float()
        mistake_ratio = mistakes / float(total)

    return loss, acc, mistake_ratio, precision, recall, f1

  def forward(self, x_tokens: torch.LongTensor) -> torch.FloatTensor:
    return self.net(x_tokens)

  def _dispatch_labels(self, x0: torch.LongTensor):
    if self.training_strategy == 'random_corruption':
      return self._compute_random_corruption_labels(x0)
    elif self.training_strategy == 'ar_perplexity':
      return self._compute_ar_perplexity_labels(x0)
    else:
      return self._compute_xt_xs_and_labels(x0)

  def training_step(self, batch, batch_idx):
    x0 = batch['input_ids'].to(self.device)

    xt, xs, new_mask, target = self._dispatch_labels(x0)
    logits = self.forward(xs)

    loss, acc, mistake_ratio, precision, recall, f1 = self._compute_loss_and_metrics(logits, target, new_mask)

    self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
    self.log('train/acc', acc, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
    self.log('train/denoiser_mistake_ratio', mistake_ratio, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
    self.log('train/precision', precision, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
    self.log('train/recall', recall, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
    self.log('train/f1', f1, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
    return loss

  def validation_step(self, batch, batch_idx):
    x0 = batch['input_ids'].to(self.device)

    xt, xs, new_mask, target = self._dispatch_labels(x0)
    logits = self.forward(xs)

    loss, acc, mistake_ratio, precision, recall, f1 = self._compute_loss_and_metrics(logits, target, new_mask)

    self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
    self.log('val/acc', acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
    self.log('val/denoiser_mistake_ratio', mistake_ratio, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
    self.log('val/precision', precision, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
    self.log('val/recall', recall, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
    self.log('val/f1', f1, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
      self.parameters(),
      lr=self.lr,
      betas=(self.config.optim.beta1, self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay,
    )
    scheduler = hydra.utils.instantiate(self.config.lr_scheduler, optimizer=optimizer)
    return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'name': 'trainer/lr'}]


def _load_denoiser(config: omegaconf.DictConfig, tokenizer) -> diffusion_mod.Diffusion:
  if 'hf' in config.backbone:
    return diffusion_mod.Diffusion(config, tokenizer=tokenizer).to('cuda')
  # Shim for old pickles that reference the long-gone
  # `transformers.models.gpt2.tokenization_gpt2_fast` module.
  import sys
  import types
  import importlib
  legacy = 'transformers.models.gpt2.tokenization_gpt2_fast'
  if legacy not in sys.modules:
    current = importlib.import_module('transformers.models.gpt2.tokenization_gpt2')
    shim = types.ModuleType(legacy)
    shim.GPT2TokenizerFast = current.GPT2Tokenizer
    sys.modules[legacy] = shim
  return diffusion_mod.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config,
    strict=False,
    weights_only=False,
  )


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config: omegaconf.DictConfig):
  _print_config(config, resolve=True, save_cfg=True)

  tokenizer = dataloader.get_tokenizer(config)
  train_loader, valid_loader = dataloader.get_dataloaders(config, tokenizer)

  assert config.eval.checkpoint_path, 'Please set eval.checkpoint_path to the denoiser checkpoint.'
  denoiser = _load_denoiser(config, tokenizer)
  denoiser.eval()
  for p in denoiser.parameters():
    p.requires_grad = False

  module = RemaskerModule(config=config, tokenizer=tokenizer, denoiser=denoiser)
  # Initialize the remasker's DIT from the denoiser's DIT weights.
  missing_keys, unexpected_keys = module.net.dit.load_state_dict(
    denoiser.backbone.state_dict(), strict=False)
  print(f"[remasker] missing keys: {missing_keys}")
  print(f"[remasker] unexpected keys: {unexpected_keys}")
  module.net.change_final_layer()

  if config.sampling.freeze_backbone:
    module.net.freeze_backbone()

  resume_ckpt_path = None
  if config.checkpointing.resume_from_ckpt:
    candidate_path = str(config.checkpointing.resume_ckpt_path)
    if candidate_path == 'last' or os.path.isfile(candidate_path):
      resume_ckpt_path = candidate_path
    else:
      print(f"[remasker] Resume requested but checkpoint not found at {candidate_path}. Starting from scratch.")

  wandb_logger = WandbLogger(
    project=config.wandb.project,
    name=f"remasker_{config.wandb.name}",
    group=config.wandb.group,
    job_type="remasker_training",
    tags=list(config.wandb.tags) + ["remasker"],
    notes=f"Remasker training - {config.wandb.notes}",
    id=f"remasker_{config.wandb.id}",
    save_dir=config.checkpointing.save_dir,
    resume='allow' if resume_ckpt_path else None,
  )

  lr_monitor = LearningRateMonitor(logging_interval='step')
  callbacks = [lr_monitor]
  if hasattr(config, 'callbacks') and config.callbacks:
    callbacks.extend([hydra.utils.instantiate(cb) for cb in config.callbacks.values()])

  trainer: L.Trainer = hydra.utils.instantiate(config.trainer, logger=wandb_logger, callbacks=callbacks)

  if resume_ckpt_path:
    print(f"[remasker] Resuming training from checkpoint: {resume_ckpt_path}")
  trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=valid_loader, ckpt_path=resume_ckpt_path)

  ckpt_dir = os.path.join(config.checkpointing.save_dir, 'checkpoints_remasker')
  os.makedirs(ckpt_dir, exist_ok=True)
  trainer.save_checkpoint(os.path.join(ckpt_dir, 'last.ckpt'))


if __name__ == '__main__':
  main()
