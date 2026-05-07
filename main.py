import os
import json

# Set tokenizer parallelism to false to avoid warnings in multiprocessing
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
from tqdm import tqdm
import numpy as np
import random
from collections import defaultdict
import multiprocessing as mp

import dataloader
import diffusion
import utils
from safetensors.torch import load_file


omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def _lazy_get_spacy_tokenizer():
  """Return a spaCy tokenizer if available, otherwise None.

  Loaded lazily to avoid startup overhead and hard dependency on spaCy model.
  """
  try:
    import spacy  # type: ignore
    # Load small English model; if unavailable, this will raise and we'll fallback
    nlp = spacy.load("en_core_web_sm")
    return nlp.tokenizer
  except Exception:
    return None


def _iter_ngrams(tokens, n):
  """Yield n-grams from a token sequence without external deps."""
  if n <= 0:
    return
  length = len(tokens)
  if length < n:
    return
  for idx in range(length - n + 1):
    yield tuple(tokens[idx: idx + n])


def compute_diversity(all_texts_list):
  """Compute n-gram repetition and aggregate diversity metrics for generated texts.

  Returns a dict with keys like '2gram_repetition', '3gram_repetition', '4gram_repetition',
  and 'diversity' which aggregates across n-gram levels.
  """
  ngram_range = (2, 3, 4)

  tokenizer = _lazy_get_spacy_tokenizer()
  token_lists = []
  for sentence in all_texts_list:
    if tokenizer is not None:
      # Use spaCy tokenizer if available
      tokens = [str(token) for token in tokenizer(sentence)]
    else:
      # Fallback: simple whitespace tokenization
      tokens = sentence.split()
    token_lists.append(tokens)

  ngram_unique_sets = {n: set() for n in ngram_range}
  ngram_total_counts = defaultdict(int)

  metrics = {}
  for n in ngram_range:
    for tokens in token_lists:
      ngrams_for_tokens = list(_iter_ngrams(tokens, n))
      if not ngrams_for_tokens:
        continue
      ngram_unique_sets[n].update(ngrams_for_tokens)
      ngram_total_counts[n] += len(ngrams_for_tokens)

    total = ngram_total_counts[n]
    unique = len(ngram_unique_sets[n])
    if total == 0:
      repetition = 0.0
    else:
      repetition = 1.0 - (unique / float(total))
    metrics[f"{n}gram_repetition"] = repetition

  diversity_product = 1.0
  for n in ngram_range:
    repetition = metrics.get(f"{n}gram_repetition", 0.0)
    diversity_component = 1.0 - repetition
    diversity_product *= diversity_component
  metrics["diversity"] = diversity_product

  return metrics


def _sampling_worker(rank,
                     config_container,
                     steps,
                     num_batches,
                     disable_ema,
                     out_queue):
  """GPU worker that generates text samples.

  Args:
    rank: CUDA device index to use in this worker.
    config_container: Serializable config container (from OmegaConf.to_container).
    steps: Diffusion sampling steps.
    num_batches: How many batches to sample in this worker.
    disable_ema: If True, disable EMA before sampling.
    out_queue: Multiprocessing queue to push results (rank, texts | error_dict).
  """
  try:
    import torch as _torch
    import omegaconf as _oc

    if _torch.cuda.is_available():
      try:
        _torch.cuda.set_device(rank)
      except Exception:
        pass

    _config = _oc.OmegaConf.create(config_container)

    _tokenizer = dataloader.get_tokenizer(_config)
    _device = _torch.device(f'cuda:{rank}')
    _model = _load_from_checkpoint(config=_config, tokenizer=_tokenizer, device=_device)
    if disable_ema:
      _model.ema = None
    _model = _model.to(_device)

    local_text_samples = []

    for _ in tqdm(range(num_batches), desc=f"Sampling on GPU {rank}"):
      samples = _model.restore_model_and_sample(num_steps=steps)
      text_samples = _model.tokenizer.batch_decode(samples)
      local_text_samples.extend(text_samples)

    out_queue.put((rank, local_text_samples))
  except Exception as _e:
    import traceback as _tb
    out_queue.put((rank, {'error': f"{_e.__class__.__name__}: {_e}", 'traceback': _tb.format_exc()}))

def _load_from_checkpoint(config, tokenizer, device=None):
  if 'hf' in config.backbone:
    return diffusion.Diffusion(
      config, tokenizer=tokenizer).to('cuda')
  
  import sys, types, importlib
  _legacy = 'transformers.models.gpt2.tokenization_gpt2_fast'
  if _legacy not in sys.modules:
    _current = importlib.import_module('transformers.models.gpt2.tokenization_gpt2')
    shim = types.ModuleType(_legacy)
    shim.GPT2TokenizerFast = _current.GPT2Tokenizer
    sys.modules[_legacy] = shim

  import warnings
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    return diffusion.Diffusion.load_from_checkpoint(
      config.eval.checkpoint_path,
      tokenizer=tokenizer,
      config=config, strict=False, device=device,
      weights_only=False)


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)


def _get_validation_texts(config, tokenizer, max_samples, seed=42):
  """Get validation texts, subsampled to match generated samples count."""
  # Fix seed for reproducibility
  np.random.seed(seed)
  random.seed(seed)
  torch.manual_seed(seed)
  
  # Get validation dataloader
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=seed)
  
  # Collect validation texts
  validation_texts = []
  for batch in tqdm(valid_ds, desc="Collecting validation texts"):
    texts = tokenizer.batch_decode(batch['input_ids']) #, skip_special_tokens=True)
    validation_texts.extend(texts)
    # Stop early if we have enough samples for efficiency
    if len(validation_texts) >= max_samples * 2:  # Collect more to subsample from
      break
  
  # Subsample to max_samples
  if len(validation_texts) > max_samples:
    validation_texts = np.random.choice(validation_texts, size=max_samples, replace=False).tolist()
  
  return validation_texts


def _compute_mauve(generated_texts, reference_texts):
  """Compute MAUVE metric. `mauve` is imported lazily to keep import-time light."""
  try:
    import mauve  # type: ignore
  except ImportError:
    print("MAUVE package is not installed. Skipping MAUVE computation.")
    return None

  result = mauve.compute_mauve(
    p_text=reference_texts,
    q_text=generated_texts,
    device_id=0 if torch.cuda.is_available() else -1,
    verbose=False,
  )
  return result.mauve


def compute_number_of_parameters(module):
  return sum(p.numel() for p in module.parameters())


def clean_sample(sample: str) -> str:
  """Clean a sample by removing all <|endoftext|> and [PAD] tokens."""
  return sample.replace('[PAD]', '').strip()


def generate_samples(config, logger, tokenizer):
  logger.info('Generating samples.')
  model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
  model.gen_ppl_metric.reset()
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  all_text_samples = []

  num_gpus = torch.cuda.device_count()
  parallel_ok = num_gpus > 1
  if parallel_ok:
    logger.info(f"Parallel sampling enabled across {num_gpus} GPUs.")
    total_batches = int(config.sampling.num_sample_batches)
    per_gpu = total_batches // num_gpus
    remainder = total_batches % num_gpus
    batches_per_rank = [per_gpu + (1 if i < remainder else 0) for i in range(num_gpus)]

    cfg_container = omegaconf.OmegaConf.to_container(config, resolve=True)

    # Ensure a CUDA-safe multiprocessing start method.
    try:
      if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
      pass

    result_q: mp.Queue = mp.Queue()
    procs = []
    for rank, nb in enumerate(batches_per_rank):
      if nb == 0:
        continue
      p = mp.Process(
        target=_sampling_worker,
        args=(rank, cfg_container, config.sampling.steps, nb,
              config.eval.disable_ema, result_q),
        daemon=True)
      p.start()
      procs.append(p)

    received = 0
    expected = sum(1 for nb in batches_per_rank if nb > 0)
    worker_errors = []
    while received < expected:
      rank, payload = result_q.get()
      if isinstance(payload, dict) and 'error' in payload:
        worker_errors.append((rank, payload['error']))
      else:
        all_text_samples.extend(payload)
      received += 1

    for p in procs:
      p.join()

    if worker_errors:
      logger.warning(f"Some sampling workers failed: {worker_errors}")

    if len(all_text_samples) > 0:
      model.compute_generative_perplexity(all_text_samples)
  else:
    for _ in tqdm(range(config.sampling.num_sample_batches)):
      samples = model.restore_model_and_sample(num_steps=config.sampling.steps)
      text_samples = model.tokenizer.batch_decode(samples)
      all_text_samples.extend(text_samples)
      model.compute_generative_perplexity(text_samples)

  validation_texts = _get_validation_texts(config, tokenizer, max_samples=len(all_text_samples))
  logger.info(f'Collected {len(validation_texts)} validation texts.')

  all_text_samples = [clean_sample(sample) for sample in all_text_samples]
  validation_texts = [clean_sample(text) for text in validation_texts]

  generated_dir = 'generated_samples'
  validation_dir = 'validation_samples'
  os.makedirs(generated_dir, exist_ok=True)
  os.makedirs(validation_dir, exist_ok=True)

  for idx, text in enumerate(all_text_samples):
    with open(os.path.join(generated_dir, f'sample_{idx}.txt'), 'w') as f:
      f.write(text)

  for idx, text in enumerate(validation_texts):
    with open(os.path.join(validation_dir, f'sample_{idx}.txt'), 'w') as f:
      f.write(text)

  gen_ppl = model.gen_ppl_metric.compute().item()
  print(f'Generative perplexity: {gen_ppl:.2f}')

  logger.info('Computing generative perplexity for validation texts...')
  batch_size = 8
  val_ppl_list = []
  for start in tqdm(range(0, len(validation_texts), batch_size), desc='Validation PPL (batch=8)'):
    batch_texts = validation_texts[start:start + batch_size]
    batch_ppl = model._compute_texts_perplexity(batch_texts, max_length=int(config.model.length))
    val_ppl_list.extend(batch_ppl)
  if len(val_ppl_list) > 0:
    val_gen_ppl = float(np.mean(val_ppl_list))
    print(f'Validation generative perplexity: {val_gen_ppl:.2f}')
  else:
    print('Validation generative perplexity: N/A')

  logger.info('Computing MAUVE metric...')
  mauve_score = _compute_mauve(generated_texts=all_text_samples, reference_texts=validation_texts)
  if mauve_score is not None:
    print(f'MAUVE score: {mauve_score * 100:.4f}')

  logger.info('Computing diversity metrics...')
  diversity_metrics = compute_diversity(all_text_samples)
  print('Diversity metrics:')
  for key in sorted(diversity_metrics.keys()):
    value = diversity_metrics[key]
    if isinstance(value, float):
      print(f'  {key}: {value * 100:.6f}')
    else:
      print(f'  {key}: {value * 100}')

  logger.info('Computing diversity metrics for validation texts...')
  validation_diversity_metrics = compute_diversity(validation_texts)
  print('Validation diversity metrics:')
  for key in sorted(validation_diversity_metrics.keys()):
    value = validation_diversity_metrics[key]
    if isinstance(value, float):
      print(f'  {key}: {value * 100:.6f}')
    else:
      print(f'  {key}: {value * 100}')

  logger.info(f'Number of parameters: {compute_number_of_parameters(model):,}')
  return mauve_score, diversity_metrics['diversity'], gen_ppl


def _ppl_eval(config, logger, tokenizer):
  logger.info('Starting Zero Shot Eval.')

  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=config.seed)
  trainer.validate(model, valid_ds)


def _train(config, logger, tokenizer):
  logger.info('Starting Training.')
  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  train_ds, valid_ds = dataloader.get_dataloaders(
    config, tokenizer)
  _print_batch(train_ds, valid_ds, tokenizer)

  model = diffusion.Diffusion(
    config, tokenizer=valid_ds.tokenizer)

  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)

  state_dict = None
  if ckpt_path:
    if ckpt_path.endswith('ckpt'):
      print(f"Loading checkpoint from {ckpt_path}")
      state_dict = torch.load(ckpt_path)["state_dict"]
    elif ckpt_path.endswith('safetensors'):
      print(f"Loading safetensors from {ckpt_path}")
      state_dict = load_file(ckpt_path)
    else:
      raise ValueError(f"Unknown checkpoint format for {ckpt_path}")
    model.load_state_dict(state_dict, strict=False)
  else:
    print('Training from scratch')
  trainer.fit(model, train_ds, valid_ds)


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)

  if config.mode == 'sample_eval':
    mauve_score, diversity, gen_ppl = generate_samples(config, logger, tokenizer)
    logger.info(f"MAUVE score: {mauve_score * 100:.4f}")
    logger.info(f"Diversity: {diversity * 100:.4f}")
    logger.info(f"Generative perplexity: {gen_ppl:.2f}")

    # Append metrics to a JSONL file. Prefer eval.metrics_file if provided.
    metrics_file = None
    if 'eval' in config and isinstance(config.eval, omegaconf.DictConfig):
      metrics_file = config.eval.get('metrics_file', None)
    if metrics_file is None or metrics_file == '':
      metrics_file = os.path.join(os.path.dirname(__file__), 'metrics.jsonl')

    metrics_dir = os.path.dirname(metrics_file) or '.'
    os.makedirs(metrics_dir, exist_ok=True)
    if not os.path.exists(metrics_file):
      with open(metrics_file, 'w') as f:
        pass  # create the file if it doesn't exist

    with open(metrics_file, 'a') as f:
      f.write(json.dumps({
        'mauve': mauve_score,
        'diversity': diversity,
        'gen_ppl': gen_ppl,
        'seed': config.seed,
        'checkpoint': config.eval.checkpoint_path
      }) + '\n')
  elif config.mode == 'ppl_eval':
    _ppl_eval(config, logger, tokenizer)
  else:
    _train(config, logger, tokenizer)


if __name__ == '__main__':
  main()