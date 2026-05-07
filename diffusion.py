import itertools
import math
import os
import json
import typing
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import transformers
from torch import Tensor

import dataloader
import models
import noise_schedule
import samplers as _samplers
import utils
from models.remasker import RemaskerNet

LOG2 = math.log(2)


def _sample_categorical(categorical_probs):
  # categorical_probs = categorical_probs.to(torch.float64)
  gumbel_norm = (
    1e-10
    - (torch.rand_like(categorical_probs) + 1e-10).log())
  return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _unsqueeze(x, reference):
  return x.view(
    * x.shape,
    * ((1,) * (len(reference.shape) - len(x.shape))))


@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor


class NLL(torchmetrics.aggregation.MeanMetric):
  pass


class BPD(NLL):
  def compute(self) -> Tensor:
    """Computes the bits per dimension.

    Returns:
      bpd
    """
    return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
  def compute(self) -> Tensor:
    """Computes the Perplexity.

    Returns:
     Perplexity
    """
    return torch.exp(self.mean_value / self.weight)


class Diffusion(L.LightningModule):
  def __init__(
    self,
    config,
    tokenizer: transformers.PreTrainedTokenizer):
    super().__init__()
    self.save_hyperparameters()
    self.config = config

    self.tokenizer = tokenizer
    self.vocab_size = len(self.tokenizer)
    self.sampler = self.config.sampling.predictor
    self.gen_ppl_eval_model_name_or_path = self.config.eval.\
      gen_ppl_eval_model_name_or_path
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.importance_sampling = self.config.training.importance_sampling
    self.change_of_variables = self.config.training.change_of_variables

    if (not hasattr(self.tokenizer, 'mask_token')
        or self.tokenizer.mask_token is None):
      self.mask_index = self.vocab_size
      self.vocab_size += 1
    else:
      self.mask_index = self.tokenizer.mask_token_id
    self.parameterization = self.config.parameterization
    if self.config.backbone == 'dit':
      self.backbone = models.dit.DIT(
        self.config, vocab_size=self.vocab_size)
    elif self.config.backbone == 'dimamba':
      self.backbone = models.dimamba.DiMamba(
        self.config,
        vocab_size=self.vocab_size,
        pad_token_id=self.tokenizer.pad_token_id)
    elif self.config.backbone == 'ar':
      self.backbone = models.autoregressive.AR(
        self.config,
        vocab_size=self.vocab_size,
        mask_index=self.mask_index)
    elif self.config.backbone == 'hf_dit':
      self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
        config.eval.checkpoint_path, trust_remote_code=True)
    else:
      raise ValueError(
        f'Unknown backbone: {self.config.backbone}')

    self.T = self.config.T
    self.subs_masking = self.config.subs_masking

    self.softplus = torch.nn.Softplus()
    # metrics are automatically reset at end of epoch
    metrics = torchmetrics.MetricCollection({
      'nll': NLL(),
      'bpd': BPD(),
      'ppl': Perplexity(),
    })
    metrics.set_dtype(torch.float64)
    self.train_metrics = metrics.clone(prefix='train/')
    self.valid_metrics = metrics.clone(prefix='val/')
    self.test_metrics = metrics.clone(prefix='test/')

    # generative perplexity
    self.gen_ppl_metric = Perplexity()
    self.eval_model_tokenizer = transformers.AutoTokenizer.\
      from_pretrained(self.gen_ppl_eval_model_name_or_path)
    if self.eval_model_tokenizer.pad_token is None:
      self.eval_model_tokenizer.pad_token =\
          self.eval_model_tokenizer.eos_token
      self.eval_model_tokenizer.pad_token_id =\
          self.eval_model_tokenizer.eos_token_id

    # Lazy-loaded AR eval model for on-the-fly PPL of x0 samples
    self._eval_ppl_model = None

    self.noise = noise_schedule.get_noise(self.config,
                                          dtype=self.dtype)

    # Optional: learned error predictor g_phi (the "remasker") for guided sampling.
    self.remasker = None
    self.remasker_t_off = float(self.config.sampling.remasker_t_off)
    self.remasker_t_on = float(self.config.sampling.remasker_t_on)
    assert self.remasker_t_off <= self.remasker_t_on, \
      "remasker_t_off must be <= remasker_t_on"
    if (self.config.sampling.predictor in ['remasker', 'star_shape']
        and self.config.sampling.remasker_checkpoint_path is not None):
      self.remasker_temperature = float(self.config.sampling.remasker_temperature)
      ckpt_path = str(self.config.sampling.remasker_checkpoint_path)
      if len(ckpt_path) == 0 or not os.path.exists(ckpt_path):
        raise ValueError(f'Remasker checkpoint path {ckpt_path} does not exist')
      print(f"Loading remasker checkpoint from {ckpt_path}")

      # Load the Lightning RemaskerModule checkpoint and strip the `net.` prefix.
      module_state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)['state_dict']
      net_state_dict = {k[len('net.'):]: v for k, v in module_state_dict.items() if k.startswith('net.')}

      self.remasker = RemaskerNet(vocab_size=self.vocab_size, config=self.config)
      self.remasker.change_final_layer()
      self.remasker.load_state_dict(net_state_dict)

      for p in self.remasker.parameters():
        p.requires_grad = False
      self.remasker.eval()
      
    if self.config.training.ema > 0:
      self.ema = models.ema.ExponentialMovingAverage(
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()),
        decay=self.config.training.ema)
    else:
      self.ema = None
    
    self.lr = self.config.optim.lr
    self.sampling_eps = self.config.training.sampling_eps
    self.time_conditioning = self.config.time_conditioning
    self.neg_infinity = -1000000.0
    self.fast_forward_epochs = None
    self.fast_forward_batches = None
    self._validate_configuration()
    # Counter for how many sampling trajectories have been saved so far
    self._trajectories_saved = 0

  def _validate_configuration(self):
    assert not (self.change_of_variables
                and self.importance_sampling)
    if self.parameterization == 'sedd':
      assert not self.importance_sampling
      assert not self.change_of_variables
    if self.parameterization == 'd3pm':
      assert self.T > 0
    if self.T > 0:
      assert self.parameterization in {'d3pm', 'subs'}
    if self.subs_masking:
      assert self.parameterization == 'd3pm'

  def on_load_checkpoint(self, checkpoint):
    if self.ema:
      self.ema.load_state_dict(checkpoint['ema'])
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
    self.fast_forward_epochs = checkpoint['loops'][
      'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
      'fit_loop']['epoch_loop.batch_progress'][
        'current']['completed']

  def on_save_checkpoint(self, checkpoint):
    if self.ema:
      checkpoint['ema'] = self.ema.state_dict()
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
    # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
    # behind, so we're using the optimizer's progress.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['total'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total'][
              'completed'] * self.trainer.accumulate_grad_batches
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['current'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['current'][
              'completed'] * self.trainer.accumulate_grad_batches
    # _batches_that_stepped tracks the number of global steps, not the number
    # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.state_dict'][
        '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']
    if 'sampler' not in checkpoint.keys():
      checkpoint['sampler'] = {}
    if hasattr(self.trainer.train_dataloader.sampler,
               'state_dict'):
      sampler_state_dict = self.trainer.\
        train_dataloader.sampler.state_dict()
      checkpoint['sampler'][
        'random_state'] = sampler_state_dict.get(
          'random_state', None)
    else:
      checkpoint['sampler']['random_state'] = None

  def on_train_start(self):
    if self.ema:
      self.ema.move_shadow_params_to_device(self.device)

    # Adapted from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
    distributed = (
      self.trainer._accelerator_connector.use_distributed_sampler
      and self.trainer._accelerator_connector.is_distributed)
    if distributed:
      sampler_cls = dataloader.FaultTolerantDistributedSampler
    else:
      sampler_cls = dataloader.RandomFaultTolerantSampler
    updated_dls = []
    for dl in self.trainer.fit_loop._combined_loader.flattened:
      if hasattr(dl.sampler, 'shuffle'):
        dl_sampler = sampler_cls(
          dl.dataset, shuffle=dl.sampler.shuffle)
      else:
        dl_sampler = sampler_cls(dl.dataset)
      if (distributed
          and self.fast_forward_epochs is not None
          and self.fast_forward_batches is not None):
        dl_sampler.load_state_dict({
          'epoch': self.fast_forward_epochs,
          'counter': (self.fast_forward_batches
                      * self.config.loader.batch_size)})
      updated_dls.append(
        torch.utils.data.DataLoader(
          dl.dataset,
          batch_size=self.config.loader.batch_size,
          num_workers=self.config.loader.num_workers,
          pin_memory=self.config.loader.pin_memory,
          sampler=dl_sampler,
          shuffle=False,
          persistent_workers=True))
    self.trainer.fit_loop._combined_loader.flattened = updated_dls

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema:
      self.ema.update(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))

  def _subs_parameterization(self, logits, xt):
    # log prob at the mask index = - infinity
    logits[:, :, self.mask_index] += self.neg_infinity
    
    # Normalize the logits such that x.exp() is
    # a probability distribution over vocab_size.
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)

    # Apply updates directly in the logits matrix.
    # For the logits of the unmasked tokens, set all values
    # to -infinity except for the indices corresponding to
    # the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)
    logits[unmasked_indices] = self.neg_infinity
    logits[unmasked_indices, xt[unmasked_indices]] = 0
    return logits

  def _d3pm_parameterization(self, logits):
    if self.subs_masking:
      logits[:, :, self.mask_index] += self.neg_infinity
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)
    return logits

  def _sedd_parameterization(self, logits, xt, sigma):
    esigm1_log = torch.where(
      sigma < 0.5,
      torch.expm1(sigma),
      sigma.exp() - 1).log().to(logits.dtype)
    # logits shape
    # (batch_size, diffusion_model_input_length, vocab_size)
    logits = logits - esigm1_log[:, None, None] - np.log(
      logits.shape[-1] - 1)
    # The below scatter operation sets the log score
    # for the input word to 0.
    logits = torch.scatter(logits, -1, xt[..., None],
                           torch.zeros_like(logits[..., :1]))
    return logits

  def _process_sigma(self, sigma):
    if sigma is None:
      assert self.parameterization == 'ar'
      return sigma
    if sigma.ndim > 1:
      sigma = sigma.squeeze(-1)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

  def forward(self, x, sigma):
    """Returns log score."""
    sigma = self._process_sigma(sigma)
    with torch.cuda.amp.autocast(dtype=torch.float32):
      logits = self.backbone(x, sigma)

    if self.parameterization == 'subs':
      return self._subs_parameterization(logits=logits,
                                         xt=x)
    elif self.parameterization == 'sedd':
      return self._sedd_parameterization(logits=logits,
                                         xt=x,
                                         sigma=sigma)
    elif self.parameterization == 'd3pm':
      return self._d3pm_parameterization(logits=logits)
    return logits

  def _d3pm_loss(self, model_output, xt, x0, t):
    dt = 1 / self.T

    if torch.is_tensor(t):
      t = t[:, None]
      assert t.ndim == 2
      t = t.clamp(0., 1. - 1e-4)
    alpha_t = 1 - t + torch.zeros_like(xt)
    alpha_s = 1 - (t - dt) + torch.zeros_like(xt)

    log_x_theta_at_x0 = torch.gather(
      model_output, -1, x0[:, :, None]).squeeze(-1)
    log_x_theta_at_m = model_output[:, :, self.mask_index]
    x_theta_at_m = log_x_theta_at_m.exp()
    
    term_1_coef = dt / t
    term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
    term_1_log_dr = log_x_theta_at_x0
    
    term_2_coef = 1 - dt / t
    term_2_log_nr = term_1_log_nr
    term_2_log_dr = torch.log(alpha_s * x_theta_at_m / (t - dt) + 1)

    L_vb_masked = (
      term_1_coef * (term_1_log_nr - term_1_log_dr)
      + term_2_coef * (term_2_log_nr - term_2_log_dr))

    L_vb = L_vb_masked * (xt == self.mask_index)

    return self.T * L_vb

  def _compute_loss(self, batch, prefix):
    if 'attention_mask' in batch:
      attention_mask = batch['attention_mask']
    else:
      attention_mask = None
    losses = self._loss(batch['input_ids'], attention_mask)
    loss = losses.loss

    if prefix == 'train':
      self.train_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.train_metrics
    elif prefix == 'val':
      self.valid_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.valid_metrics
    elif prefix == 'test':
      self.test_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.test_metrics
    else:
      raise ValueError(f'Invalid prefix: {prefix}')

    self.log_dict(metrics,
                  on_step=False,
                  on_epoch=True,
                  sync_dist=True)
    return loss

  def on_train_epoch_start(self):
    self.backbone.train()
    self.noise.train()

  def training_step(self, batch, batch_idx):
    loss = self._compute_loss(batch, prefix='train')
    self.log(name='trainer/loss',
             value=loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return loss

  def on_validation_epoch_start(self):
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))

    self.backbone.eval()
    self.noise.eval()
    assert self.valid_metrics.nll.mean_value == 0
    assert self.valid_metrics.nll.weight == 0

  def validation_step(self, batch, batch_idx):
    return self._compute_loss(batch, prefix='val')

  def on_validation_epoch_end(self):
    if ((self.config.eval.compute_perplexity_on_sanity
         or not self.trainer.sanity_checking)
         and self.config.eval.generate_samples
         and not self.parameterization == 'ar'):
      # TODO(justin): implement sampling and kv cache for AR
      samples, text_samples = None, None
      for _ in range(
        self.config.sampling.num_sample_batches):
        samples, _ = self._sample()
        # Decode the samples to be re-tokenized by eval model
        text_samples = self.tokenizer.batch_decode(samples)
        if self.config.eval.compute_generative_perplexity:
          self.compute_generative_perplexity(text_samples)
      if self.trainer.global_rank == 0 and hasattr(
        self.trainer.logger, 'log_table'):
        # Log the last generated samples
        text_samples = text_samples[
          : self.config.sampling.num_sample_log]
        self.trainer.logger.log_table(
          key=f'samples@global_step{self.global_step}',
          columns=['Generated Samples'],
          data=[[s] for s in text_samples])
      if self.config.eval.compute_generative_perplexity:
        self.log('val/gen_ppl',
                 self.gen_ppl_metric,
                 on_epoch=True,
                 on_step=False,
                 sync_dist=True)
    if self.ema:
      self.ema.restore(
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()))

  def configure_optimizers(self):
    # TODO(yair): Lightning currently giving this warning when using `fp16`:
    #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
    #  Not clear if this is a problem or not.
    #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
    optimizer = torch.optim.AdamW(
      itertools.chain(self.backbone.parameters(),
                      self.noise.parameters()),
      lr=self.config.optim.lr,
      betas=(self.config.optim.beta1,
             self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay)

    scheduler = hydra.utils.instantiate(
      self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {
      'scheduler': scheduler,
      'interval': 'step',
      'monitor': 'val/loss',
      'name': 'trainer/lr',
    }
    return [optimizer], [scheduler_dict]

  @torch.no_grad()
  def eval_retokenize(self, text_samples, max_length):
    """Retokenizes samples for the eval model.
    
    Args:
        text_samples: List of sentences generated by the model.
    Returns:
        samples: Samples re-tokenized for the eval model
        attn_mask: Attention mask for the eval model
        eval_context_size: Size of the context for the eval model
    """
    if 'llama2' in self.gen_ppl_eval_model_name_or_path:
      tokenizer_kwargs = {
        'text_samples': text_samples,
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,
      }
      eval_context_size = 4096
    else:
      tokenizer_kwargs = {
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,
      }
      eval_context_size = 1024
    samples = self.eval_model_tokenizer(
      text_samples, ** tokenizer_kwargs)
    attn_mask = samples['attention_mask']
    samples = samples['input_ids']
    if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
      attn_mask = attn_mask.to(self.device)
      samples = samples.to(self.device)      
    return samples, attn_mask, eval_context_size

  @torch.no_grad()
  def compute_generative_perplexity(
    self,
    text_samples: typing.List[str],
    retokenize: bool = True,
    max_length: typing.Optional[int] = None) -> None:
    """Compute the generative perplexity of the model.

    Args:
        text_samples: List of sentences generated by the model.
    
    Returns:
        Perplexity of the generated text under a different
        pre-trained AR model (e.g., GPT2).
    """
    # Lazily load and cache the eval AR model
    if self._eval_ppl_model is None:
      os.environ['TOKENIZERS_PARALLELISM'] = 'false'
      eval_model = transformers.AutoModelForCausalLM.from_pretrained(
        self.gen_ppl_eval_model_name_or_path).eval()
      if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
        eval_model = eval_model.to(self.device)
      self._eval_ppl_model = eval_model
    eval_model = self._eval_ppl_model
    if max_length is None:
      max_length = self.config.model.length
    # Re-tokenize using eval model's tokenizer
    if retokenize:
      (samples, attn_mask,
       eval_context_size) = self.eval_retokenize(
         text_samples, max_length=max_length)
    else:
      samples = text_samples
      attn_mask = torch.ones(samples.shape).to(self.device)
      eval_context_size = samples.shape[-1]
    batch_size = min(
      self.config.eval.perplexity_batch_size,
      samples.shape[0])
    num_batches = samples.shape[0] // batch_size
    for i in range(num_batches):
      _samples = torch.split(
        samples[i * batch_size: (i + 1) * batch_size],
        eval_context_size,
        dim=-1)
      _attn_mask = torch.split(
        attn_mask[i * batch_size: (i + 1) * batch_size],
        eval_context_size,
        dim=-1)
      for (sample_chunk, attn_mask_chunk) in zip(
        _samples, _attn_mask):
        logits = eval_model(
          sample_chunk, attention_mask=attn_mask_chunk)[0]
        logits = logits.transpose(-1, -2)
        
        nlls = F.cross_entropy(logits[..., :-1],
                               sample_chunk[..., 1:],
                               reduction='none')
        first_eos = (sample_chunk == self.eval_model_tokenizer\
                     .eos_token_id).cumsum(-1) == 1
        token_mask = (
          sample_chunk
          != self.eval_model_tokenizer.eos_token_id)
        self.gen_ppl_metric.update(
          nlls, first_eos[..., 1:] + token_mask[..., 1:])

  @torch.no_grad()
  def _compute_texts_perplexity(
    self,
    text_samples: typing.List[str],
    max_length: typing.Optional[int] = None,
  ) -> typing.List[float]:
    """Return per-text generative perplexity using the configured AR eval model.

    This mirrors compute_generative_perplexity but returns per-sample PPLs
    instead of updating the running metric.
    """
    # Lazily materialize the eval AR model once
    if self._eval_ppl_model is None:
      os.environ['TOKENIZERS_PARALLELISM'] = 'false'
      eval_model = transformers.AutoModelForCausalLM.from_pretrained(
        self.gen_ppl_eval_model_name_or_path).eval()
      if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
        eval_model = eval_model.to(self.device)
      self._eval_ppl_model = eval_model

    if max_length is None:
      max_length = int(self.config.model.length)

    # Re-tokenize using eval model's tokenizer to ensure vocab alignment
    samples, attn_mask, _ = self.eval_retokenize(text_samples, max_length=max_length)

    # Forward once; shapes: logits [B, T, V]
    outputs = self._eval_ppl_model(samples, attention_mask=attn_mask)
    logits = outputs[0].transpose(-1, -2)  # -> [B, V, T]

    # Next-token prediction loss
    nlls = F.cross_entropy(
      logits[..., :-1],  # [B, V, T-1]
      samples[..., 1:],  # [B, T-1]
      reduction='none')  # [B, T-1]

    # Mask tokens after first EOS, but keep tokens before/at first EOS
    first_eos = (samples == self.eval_model_tokenizer.eos_token_id).cumsum(-1) == 1
    token_mask = (samples != self.eval_model_tokenizer.eos_token_id)
    mask = (first_eos[..., 1:] + token_mask[..., 1:])  # int mask [B, T-1]

    # Per-sample mean NLL -> PPL
    nll_sum = (nlls * mask).sum(dim=-1)
    count = mask.sum(dim=-1).clamp_min(1)
    mean_nll = nll_sum / count
    ppl = torch.exp(mean_nll).detach().to('cpu').tolist()
    return [float(x) for x in ppl]

  def q_xt(self, x, move_chance):
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input. 
      move_chance: float torch.Tensor with shape (batch_size, 1).
    """
    move_indices = torch.rand(
      * x.shape, device=x.device) < move_chance
    xt = torch.where(move_indices, self.mask_index, x)
    return xt

  def _sample_prior(self, *batch_dims):
    """Return a tensor of shape `batch_dims` filled with the [MASK] token id."""
    return self.mask_index * torch.ones(*batch_dims, dtype=torch.int64)

  def _ddpm_caching_update(self, x, t, dt, p_x0=None, conf=None):
    assert self.config.noise.type == 'loglinear'
    sigma_t, _ = self.noise(t)
    if t.ndim > 1:
      t = t.squeeze(-1)
    assert t.ndim == 1
    move_chance_t = t[:, None, None]
    move_chance_s = (t - dt)[:, None, None]
    assert move_chance_t.ndim == 3, move_chance_t.shape
    if p_x0 is None:
      log_p_x0 = self.forward(x, sigma_t)

      denoiser_temp = self.config.sampling.denoiser_temp_during_remasking
      denoiser_temp = max(1e-6, denoiser_temp)
      log_p_x0 = log_p_x0 / denoiser_temp
      
      if self.config.sampling.nucleus_p < 1:
        p_x0 = log_p_x0.exp()
        sorted_probs, sorted_indices = torch.sort(p_x0, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        top_p_mask = cumulative_probs <= self.config.sampling.nucleus_p
        top_p_mask[..., 0] = True
        nucleus_probs = sorted_probs * top_p_mask
        nucleus_probs /= nucleus_probs.sum(dim=-1, keepdim=True)
        p_x0 = torch.zeros_like(p_x0).scatter_(-1, sorted_indices, nucleus_probs)
      else:
        p_x0 = log_p_x0.exp()
    
    # if self.config.sampling.remdm_mode in ["cap", "rescale", "conf", "loop"] or self.config.sampling.use_fp64:
    if self.config.sampling.use_fp64:
      move_chance_t = move_chance_t.to(torch.float64)
      move_chance_s = move_chance_s.to(torch.float64)
      p_x0 = p_x0.to(torch.float64)
    
    padding_mask = x == self.tokenizer.pad_token_id

    _x = None

    assert move_chance_t.ndim == p_x0.ndim
    if self.config.sampling.remdm_mode is None:
      q_xs = p_x0 * (move_chance_t - move_chance_s)
      q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
      _x = _sample_categorical(q_xs)
      
      copy_flag = (x != self.mask_index).to(x.dtype)
      xs = copy_flag * x + (1 - copy_flag) * _x
    elif self.config.sampling.remdm_mode == "cap":
      alpha_t = (1 - move_chance_t)[0].item()
      alpha_s = (1 - move_chance_s)[0].item()
      if alpha_t > 0:
        sigma = min(self.config.sampling.eta, (1 - alpha_s) / alpha_t)
      else:
        sigma = self.config.sampling.eta
      q_xs = p_x0 * (1 - sigma)
      q_xs[..., self.mask_index] = sigma
      q_xs_2 = p_x0 * ((alpha_s - (1 - sigma) * alpha_t) / (1 - alpha_t))
      q_xs_2[..., self.mask_index] = (1 - alpha_s - sigma * alpha_t) / (1 - alpha_t)

      copy_flag = (x != self.mask_index).to(torch.bool)
      q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
      xs = _sample_categorical(q_xs)
    elif self.config.sampling.remdm_mode == "rescale":
      alpha_t = (1 - move_chance_t)[0].item()
      alpha_s = (1 - move_chance_s)[0].item()
      if alpha_t > 0:
        sigma_max = min(1, (1 - alpha_s) / alpha_t)
      else:
        sigma_max = 1
      sigma = self.config.sampling.eta * sigma_max
      q_xs = p_x0 * (1 - sigma)
      q_xs[..., self.mask_index] = sigma
      q_xs_2 = p_x0 * ((alpha_s - (1 - sigma) * alpha_t) / (1 - alpha_t))
      q_xs_2[..., self.mask_index] = (1 - alpha_s - sigma * alpha_t) / (1 - alpha_t)
      copy_flag = (x != self.mask_index).to(torch.bool)
      q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
      xs = _sample_categorical(q_xs)
    elif self.config.sampling.remdm_mode == "conf":
      alpha_t = (1 - move_chance_t)[0].item()
      alpha_s = (1 - move_chance_s)[0].item()
      if alpha_t > 0:
        sigma_max = min(1, (1 - alpha_s) / alpha_t)
      else:
        sigma_max = 1
      eta = conf.softmax(dim=-1)
      masked_flag = (x == self.mask_index).to(torch.bool)
      eta[masked_flag] = 0
      sigma = eta * sigma_max
      q_xs = p_x0 * (1 - sigma[:, :, None])
      q_xs[..., self.mask_index] = sigma
      q_xs_2 = p_x0 * ((alpha_s - (1 - sigma[:, :, None]) * alpha_t) / (1 - alpha_t))
      q_xs_2[..., self.mask_index] = (1 - alpha_s - sigma * alpha_t) / (1 - alpha_t)
      copy_flag = (x != self.mask_index).to(torch.bool)
      q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
      xs = _sample_categorical(q_xs)
      # update conf
      unmask_mask = (x == self.mask_index) & (xs != self.mask_index)
      batch_indices = torch.arange(xs.shape[0])[:, None]
      feature_indices = torch.arange(xs.shape[1])
      conf_values = - p_x0[batch_indices, feature_indices, xs]
      conf[unmask_mask] = conf_values[unmask_mask]
      remask_mask = (x != self.mask_index) & (xs == self.mask_index)
      conf[remask_mask] = -torch.inf
    elif self.config.sampling.remdm_mode == "loop":
      time = t[0].item()
      # compute alpha_t and alpha_s
      if time > self.config.sampling.t_on:
        move_chance_t = (1 - (1 - t) * self.config.sampling.alpha_on / (1 - self.config.sampling.t_on))[:, None, None]
        move_chance_s = (1 - (1 - t + dt) * self.config.sampling.alpha_on / (1 - self.config.sampling.t_on))[:, None, None]
      elif time <= self.config.sampling.t_off:
        move_chance_t = (t * (1 - self.config.sampling.alpha_on) / self.config.sampling.t_off)[:, None, None]
        move_chance_s = ((t - dt) * (1 - self.config.sampling.alpha_on) / self.config.sampling.t_off)[:, None, None]
      else:
        move_chance_t, move_chance_s = None, None
      # use MDLM
      if time > self.config.sampling.t_on or time <= self.config.sampling.t_off:
        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        _x = _sample_categorical(q_xs)
        copy_flag = (x != self.mask_index).to(x.dtype)
        xs = copy_flag * x + (1 - copy_flag) * _x
      else: # use ReMDM
        sigma = self.config.sampling.eta
        q_xs = p_x0 * (1 - sigma)
        q_xs[..., self.mask_index] = sigma
        q_xs_2 = p_x0 * ((self.config.sampling.alpha_on - (1 - sigma) * self.config.sampling.alpha_on) / (1 - self.config.sampling.alpha_on))
        q_xs_2[..., self.mask_index] = (1 - self.config.sampling.alpha_on - self.config.sampling.alpha_on * sigma) / (1 - self.config.sampling.alpha_on)
        copy_flag = (x != self.mask_index).to(torch.bool)
        q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
        xs = _sample_categorical(q_xs)
    elif self.config.sampling.remdm_mode == "loop_star_shape":
      time = t[0].item()
      # compute alpha_t and alpha_s
      if time > self.config.sampling.t_on:
        move_chance_t = (1 - (1 - t) * self.config.sampling.alpha_on / (1 - self.config.sampling.t_on))[:, None, None]
        move_chance_s = (1 - (1 - t + dt) * self.config.sampling.alpha_on / (1 - self.config.sampling.t_on))[:, None, None]
      elif time <= self.config.sampling.t_off:
        move_chance_t = (t * (1 - self.config.sampling.alpha_on) / self.config.sampling.t_off)[:, None, None]
        move_chance_s = ((t - dt) * (1 - self.config.sampling.alpha_on) / self.config.sampling.t_off)[:, None, None]
      else:
        move_chance_t, move_chance_s = None, None
      # use MDLM
      if time > self.config.sampling.t_on or time <= self.config.sampling.t_off:
        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        _x = _sample_categorical(q_xs)
        copy_flag = (x != self.mask_index).to(x.dtype)
        xs = copy_flag * x + (1 - copy_flag) * _x
      else: # use ReMDM with sigma = 1 - alpha_s
        sigma = 1 - self.config.sampling.alpha_on
        q_xs = p_x0 * (1 - sigma)
        q_xs[..., self.mask_index] = sigma
        q_xs_2 = p_x0 * ((self.config.sampling.alpha_on - (1 - sigma) * self.config.sampling.alpha_on) / (1 - self.config.sampling.alpha_on))
        q_xs_2[..., self.mask_index] = (1 - self.config.sampling.alpha_on - self.config.sampling.alpha_on * sigma) / (1 - self.config.sampling.alpha_on)
        copy_flag = (x != self.mask_index).to(torch.bool)
        q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
        xs = _sample_categorical(q_xs)
    else:
      raise ValueError(f"Invalid remdm_mode: {self.config.sampling.remdm_mode}")

    xs[padding_mask] = self.tokenizer.pad_token_id

    return p_x0, xs, conf, _x

  def _ddpm_update(self, x, t, dt, confident_score=None):
    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)
    if sigma_t.ndim > 1:
      sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)
    assert sigma_t.ndim == 1, sigma_t.shape
    assert sigma_s.ndim == 1, sigma_s.shape
    move_chance_t = 1 - torch.exp(-sigma_t)
    move_chance_s = 1 - torch.exp(-sigma_s)
    move_chance_t = move_chance_t[:, None, None]
    move_chance_s = move_chance_s[:, None, None]
    unet_conditioning = sigma_t
    log_p_x0 = self.forward(x, unet_conditioning)
    # When remasker-guided sampling is active, scale the denoiser logits by a
    # configurable temperature before forming the proposal distribution.
    if self.sampler == "remasker":
      log_p_x0 = log_p_x0 / self.config.sampling.denoiser_temp_during_remasking
    if self.sampler == "p2":
      log_p_x0 = log_p_x0 / self.config.p2.temperature
    # applying stable softmax
    log_p_x0 = log_p_x0 - log_p_x0.max(dim=-1, keepdim=True).values
    p_x0 = log_p_x0.exp()
    p_x0 = p_x0 / p_x0.sum(dim=-1, keepdim=True)

    if self.config.sampling.nucleus_p < 1:
      sorted_probs, sorted_indices = torch.sort(p_x0, descending=True, dim=-1)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= self.config.sampling.nucleus_p
      top_p_mask[..., 0] = True
      nucleus_probs = sorted_probs * top_p_mask
      nucleus_probs /= nucleus_probs.sum(dim=-1, keepdim=True)
      p_x0 = torch.zeros_like(p_x0).scatter_(-1, sorted_indices, nucleus_probs)
    assert move_chance_t.ndim == log_p_x0.ndim
    # Technically, this isn't q_xs since there's a division
    # term that is missing. This division term doesn't affect
    # the samples.
    if self.config.sampling.use_fp64:
      p_x0 = p_x0.to(torch.float64)
      move_chance_t = move_chance_t.to(torch.float64)
      move_chance_s = move_chance_s.to(torch.float64)
    q_xs = p_x0 * (move_chance_t
                             - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    _x = _sample_categorical(q_xs)

    copy_flag = (x != self.mask_index).to(x.dtype)

    if confident_score is None:
      return copy_flag * x + (1 - copy_flag) * _x, _x
    else:
      was_masked = (x == self.mask_index)
      new_confident_score = confident_score.to(dtype=p_x0.dtype).clone()
      proposed_token_probs = p_x0.gather(-1, _x.unsqueeze(-1)).squeeze(-1)
      new_confident_score[was_masked] = -proposed_token_probs[was_masked]
      return copy_flag * x + (1 - copy_flag) * _x, new_confident_score, _x

  def _remasker_update(self, x, t, dt):
    """Thin wrapper around `samplers.remasker_update` (see that module for docs)."""
    return _samplers.remasker_update(self, x, t, dt)

  def _conf_star_shape_update(self, x, t, dt, conf):
    """Thin wrapper around `samplers.conf_star_shape_update`."""
    return _samplers.conf_star_shape_update(self, x, t, dt, conf)

  def _p2_update(self, x, t, dt):
    """Thin wrapper around `samplers.p2_update`."""
    return _samplers.p2_update(self, x, t, dt)

  @torch.no_grad()
  def _sample(self, num_steps=None, eps=1e-5):
    """Generate samples from the model (fully unconditional)."""
    batch_size_per_gpu = self.config.loader.eval_batch_size
    if num_steps is None:
      num_steps = self.config.sampling.steps

    x = self._sample_prior(
      batch_size_per_gpu,
      self.config.model.length).to(self.device)

    # Optional: setup per-step trajectory recording
    save_traj = False
    max_to_save = 0
    try:
      save_traj = bool(self.config.eval.save_sampling_trajectory)
      max_to_save = int(self.config.eval.max_trajectories_to_save)
    except Exception:
      save_traj = False
      max_to_save = 0
    remaining_to_save = max(0, max_to_save - getattr(self, '_trajectories_saved', 0))
    indices_to_save = list(range(min(remaining_to_save, x.shape[0])) if (save_traj and remaining_to_save > 0) else [])
    traj_records = {idx: [] for idx in indices_to_save}
    def _record_state(current_x, current_t, x_0_sample=None):
      if len(indices_to_save) == 0:
        return
      if isinstance(current_t, torch.Tensor):
        t_scalar = float(current_t.view(-1)[0].item())
      else:
        t_scalar = float(current_t)
      for _idx in indices_to_save:
        _xi = current_x[_idx].detach().to('cpu')
        tokens_list = _xi.to(torch.int64).tolist()
        # Robust detokenization including special tokens; insert [MASK] for our custom mask id
        tokens_str = []
        for _id in tokens_list:
          if _id == self.mask_index:
            tokens_str.append('[MASK]')
          else:
            try:
              tok = self.tokenizer.convert_ids_to_tokens([_id])[0]
            except Exception:
              tok = self.tokenizer.unk_token if hasattr(self.tokenizer, 'unk_token') and self.tokenizer.unk_token is not None else '<unk>'
            tokens_str.append(tok)
        try:
          x_text = self.tokenizer.convert_tokens_to_string(tokens_str)
        except Exception:
          x_text = ' '.join(tokens_str)
        rec = {'t': f"{t_scalar:.6f}", 'x': x_text, 'x_tokens': tokens_list}

        # Optionally attach x0_sample tokens/text and its generative perplexity
        if (x_0_sample is not None) and bool(getattr(self.config.sampling, 'save_x0_sample', False)):
          try:
            _x0 = x_0_sample[_idx].detach().to('cpu')
            x0_tokens_list = _x0.to(torch.int64).tolist()
            x0_tokens_str = []
            for _id in x0_tokens_list:
              if _id == self.mask_index:
                x0_tokens_str.append('[MASK]')
              else:
                try:
                  tok = self.tokenizer.convert_ids_to_tokens([_id])[0]
                except Exception:
                  tok = self.tokenizer.unk_token if hasattr(self.tokenizer, 'unk_token') and self.tokenizer.unk_token is not None else '<unk>'
                x0_tokens_str.append(tok)
            try:
              x0_text = self.tokenizer.convert_tokens_to_string(x0_tokens_str)
            except Exception:
              x0_text = ' '.join(x0_tokens_str)

            # Compute gen PPL for x0_text using cached eval model
            x0_ppl_list = self._compute_texts_perplexity([x0_text])
            x0_ppl = float(x0_ppl_list[0]) if len(x0_ppl_list) > 0 else None

            rec.update({'x0': x0_text, 'x0_tokens': x0_tokens_list, 'x0_ppl': x0_ppl})
          except Exception:
            # Best-effort; skip x0 details on failure
            pass

        traj_records[_idx].append(rec)
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None

    if self.config.sampling.use_fp64:
      conf_dtype = torch.float64
    else:
      conf_dtype = torch.bfloat16
    confident_score = - torch.ones_like(x, device=self.device).to(conf_dtype) * torch.inf
    x0_sample = None
    for i in range(num_steps):
      t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
      if self.sampler == 'ddpm':
        x, x0_sample = self._ddpm_update(x, t, dt)
      elif self.sampler == 'ddpm_cache':
        p_x0_cache, x_next, confident_score, x0_sample = self._ddpm_caching_update(
          x, t, dt, p_x0=p_x0_cache, conf=confident_score)
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          # Disable caching if anything changed or we need per-step sigma.
          p_x0_cache = None
        x = x_next
      elif self.sampler == 'remasker':
        # Remasker-guided update, gated by [remasker_t_off, remasker_t_on].
        p_x0_cache = None
        t_scalar = float(t.view(-1)[0].item())
        if (t_scalar >= self.remasker_t_off) and (t_scalar <= self.remasker_t_on):
          x, x0_sample = self._remasker_update(x, t, dt)
        else:
          x, x0_sample = self._ddpm_update(x, t, dt)
      elif self.sampler == 'conf_star_shape':
        # Confidence-based star-shaped update; only when t in (t_off, t_on].
        p_x0_cache = None
        t_scalar = float(t.view(-1)[0].item())
        if (t_scalar <= self.config.sampling.t_on) and (t_scalar > self.config.sampling.t_off):
          x, confident_score, x0_sample = self._conf_star_shape_update(x, t, dt, confident_score)
        else:
          x, confident_score, x0_sample = self._ddpm_update(x, t, dt, confident_score)
      elif self.sampler == "star_shape":
        # Paper: remasker inside [0, t_on], plain DDPM elsewhere.
        assert self.config.noise.type == "loglinear"
        t_on = self.config.sampling.t_on
        t_scalar = float(t.view(-1)[0].item())
        if t_scalar <= t_on:
          x, x0_sample = self._remasker_update(x, t, dt)
        else:
          x, x0_sample = self._ddpm_update(x, t, dt)
      elif self.sampler == 'p2':
        # P2 baseline, gated by [t_off, t_on].
        t_scalar = float(t.view(-1)[0].item())
        p2_t_off = float(self.config.p2.t_off)
        p2_t_on = float(self.config.p2.t_on)
        if (t_scalar >= p2_t_off) and (t_scalar <= p2_t_on):
          x, x0_sample = self._p2_update(x, t, dt)
        else:
          x, x0_sample = self._ddpm_update(x, t, dt)
      else:
        x = self._analytic_update(x, t, dt)
        x0_sample = None

      _record_state(x, t, x0_sample)

    if self.config.sampling.noise_removal:
      t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
      if self.sampler == 'analytic':
        x = self._denoiser_update(x, t)
      else:
        unet_conditioning = self.noise(t)[0]
        x = self.forward(x, unet_conditioning).argmax(dim=-1)

    # Persist recorded trajectories to disk.
    if len(indices_to_save) > 0:
      import random
      traj_dir = f'sampling_trajectories/{random.randint(0, 1000000)}'
      os.makedirs(traj_dir, exist_ok=True)
      base_id = getattr(self, '_trajectories_saved', 0)
      for j, _idx in enumerate(indices_to_save):
        traj_id = base_id + j
        jsonl_path = os.path.join(traj_dir, f'traj_{traj_id}.jsonl')
        with open(jsonl_path, 'w', encoding='utf-8') as f:
          for rec in traj_records[_idx]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
      self._trajectories_saved = base_id + len(indices_to_save)

    return x

  def restore_model_and_sample(self, num_steps, eps=1e-5):
    """Generate samples from the model."""
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    samples = self._sample(num_steps=num_steps, eps=eps)
    if self.ema:
      self.ema.restore(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.train()
    self.noise.train()
    return samples

  def get_score(self, x, sigma):
    model_output = self.forward(x, sigma)
    if self.parameterization == 'subs':
      # score(x, t) = p_t(y) / p_t(x)
      # => log score(x, t) = log p_t(y) - log p_t(x)
      
      # case 1: x = masked
      #   (i) y = unmasked
      #     log score(x, t) = log p_\theta(x)|_y + log k
      #     where k = exp(- sigma) / (1 - exp(- sigma))
      #   (ii) y = masked
      #     log score(x, t) = 0

      # case 2: x = unmasked
      #   (i) y != masked, y != x
      #     log score(x_i, t) = - inf
      #   (ii) y = x 
      #     log score(x_i, t) = 0
      #   (iii) y = masked token
      #     log score(x_i, t) = - log k
      #     where k = exp(- sigma) / (1 - exp(- sigma))
      
      log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
      assert log_k.ndim == 1
      
      masked_score = model_output + log_k[:, None, None]
      masked_score[:, :, self.mask_index] = 0

      unmasked_score = self.neg_infinity * torch.ones_like(
        model_output)
      unmasked_score = torch.scatter(
        unmasked_score,
        -1,
        x[..., None],
        torch.zeros_like(unmasked_score[..., :1]))
      unmasked_score[:, :, self.mask_index] = - (
        log_k[:, None] * torch.ones_like(x))
      
      masked_indices = (x == self.mask_index).to(
        model_output.dtype)[:, :, None]
      model_output = (
        masked_score * masked_indices
        + unmasked_score * (1 - masked_indices))
    return model_output.exp()

  def _staggered_score(self, score, dsigma):
    score = score.clone()
    extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
    score *= dsigma.exp()[:, None]
    score[..., self.mask_index] += extra_const
    return score

  def _analytic_update(self, x, t, step_size):
    curr_sigma, _ = self.noise(t)
    next_sigma, _ = self.noise(t - step_size)
    dsigma = curr_sigma - next_sigma
    score = self.get_score(x, curr_sigma)
    stag_score = self._staggered_score(score, dsigma)
    probs = stag_score * self._transp_transition(x, dsigma)
    return _sample_categorical(probs)

  def _denoiser_update(self, x, t):
    sigma, _ = self.noise(t)
    score = self.get_score(x, sigma)
    stag_score = self._staggered_score(score, sigma)
    probs = stag_score * self._transp_transition(x, sigma)
    probs[..., self.mask_index] = 0
    samples = _sample_categorical(probs)
    return samples

  def _transp_transition(self, i, sigma):
    sigma = _unsqueeze(sigma, reference=i[..., None])
    edge = torch.exp(-sigma) * F.one_hot(
      i, num_classes=self.vocab_size)
    edge += torch.where(i == self.mask_index,
                        1 - torch.exp(-sigma).squeeze(-1),
                        0)[..., None]
    return edge

  def _sample_t(self, n, device):
    _eps_t = torch.rand(n, device=device)
    if self.antithetic_sampling:
      offset = torch.arange(n, device=device) / n
      _eps_t = (_eps_t / n + offset) % 1
    t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
    if self.importance_sampling:
      return self.noise.importance_sampling_transformation(t)
    return t

  def _maybe_sub_sample(self, x0, attention_mask):
    seqlen = x0.shape[1]
    if seqlen > self.config.model.length:
      assert seqlen == 2 * self.config.model.length
      # cropping is needed for text8-crop dataset
      # try the same starting point for now
      start = np.random.choice(self.config.model.length)
      end = start + self.config.model.length
      input_tokens = x0[:, start: end]
      output_tokens = x0[:, start + 1: end + 1]
      new_attention_mask = attention_mask[:, start: end]

      # Helps with validation PPL, since the val
      # examples will all start and end with BOS/EOS
      input_tokens[:, 0] = self.tokenizer.bos_token_id
      output_tokens[:, -1] = self.tokenizer.eos_token_id
    elif self.parameterization == 'ar':
      input_tokens = x0[:, :-1]
      output_tokens = x0[:, 1:]
      new_attention_mask = attention_mask[:, 1:]
    else:
      input_tokens = x0
      output_tokens = None
      new_attention_mask = attention_mask
    return input_tokens, output_tokens, new_attention_mask

  def _reconstruction_loss(self, x0):
    t0 = torch.zeros(x0.shape[0], dtype=self.dtype,
                     device=self.device)
    assert self.config.noise.type == 'loglinear'
    unet_conditioning = self.noise(t0)[0][:, None]
    model_output_t0 = self.forward(x0, unet_conditioning)
    return - torch.gather(input=model_output_t0,
                          dim=-1,
                          index=x0[:, :, None]).squeeze(-1)

  def _forward_pass_diffusion(self, x0):
    t = self._sample_t(x0.shape[0], x0.device)
    if self.T > 0:
      t = (t * self.T).to(torch.int)
      t = t / self.T
      # t \in {1/T, 2/T, ..., 1}
      t += (1 / self.T)

    if self.change_of_variables:
      unet_conditioning = t[:, None]
      f_T = torch.log1p(- torch.exp(- self.noise.sigma_max))
      f_0 = torch.log1p(- torch.exp(- self.noise.sigma_min))
      move_chance = torch.exp(f_0 + t * (f_T - f_0))
      move_chance = move_chance[:, None]
    else:
      sigma, dsigma = self.noise(t)
      unet_conditioning = sigma[:, None]
      move_chance = 1 - torch.exp(-sigma[:, None])

    xt = self.q_xt(x0, move_chance)
    model_output = self.forward(xt, unet_conditioning)
    utils.print_nans(model_output, 'model_output')

    if self.parameterization == 'sedd':
      return dsigma[:, None] * self._score_entropy(
        model_output, sigma[:, None], xt, x0)
    
    if self.T > 0:
      diffusion_loss = self._d3pm_loss(
        model_output=model_output, xt=xt, x0=x0, t=t)
      if self.parameterization == 'd3pm':
        reconstruction_loss = self._reconstruction_loss(x0)
      elif self.parameterization == 'subs':
        reconstruction_loss = 0
      return reconstruction_loss + diffusion_loss
    
    # SUBS parameterization, continuous time.
    log_p_theta = torch.gather(
      input=model_output,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    
    if self.change_of_variables or self.importance_sampling:
      return log_p_theta * torch.log1p(
        - torch.exp(- self.noise.sigma_min))
    
    return - log_p_theta * (
      dsigma / torch.expm1(sigma))[:, None]

  def _loss(self, x0, attention_mask):
    (input_tokens, output_tokens,
     attention_mask) = self._maybe_sub_sample(
       x0, attention_mask)

    if self.parameterization == 'ar':
      logprobs = self.backbone(input_tokens, None)
      loss = - logprobs.gather(
        -1, output_tokens[:, :, None])[:, :, 0]
    else:
      loss = self._forward_pass_diffusion(input_tokens)
    
    nlls = loss * attention_mask
    count = attention_mask.sum()

    batch_nll = nlls.sum()
    token_nll = batch_nll / count

    return Loss(loss=token_nll,
                nlls=nlls,
                token_mask=attention_mask)

  def _score_entropy(self, log_score, sigma, xt, x0):
    """Computes the SEDD loss.

    Args:
      log_score: float torch.Tensor with shape (batch_size,
          diffusion_model_input_length, vocab_size),
          log score, output of the denoising network.
      xt: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      x0: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      sigma: float torch.Tensor with shape (batch_size, 1).

    Returns:
      loss with shape (batch_size, diffusion_model_input_length)
    """
    masked_indices = xt == self.mask_index

    expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
    q_ratio = 1 / expsig_minus_1[masked_indices]

    words_that_were_masked = x0[masked_indices]

    neg_term = q_ratio * torch.gather(
      log_score[masked_indices],
      -1,
      words_that_were_masked[..., None]).squeeze(-1)
    score = log_score[masked_indices].exp()
    if self.mask_index == self.vocab_size - 1:
      pos_term = score[:, :-1].sum(dim=-1)
    else:
      pos_term = score[:, : self.mask_index].sum(
        dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
    const = q_ratio * (q_ratio.log() - 1)

    entropy = torch.zeros(* xt.shape, device=xt.device)
    entropy[masked_indices] += pos_term - neg_term + const
    return entropy
