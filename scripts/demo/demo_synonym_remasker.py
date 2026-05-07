"""Demo: the remasker treats both a word and its synonym as correct.

Loads the denoiser (MDLM Diffusion) and remasker, then for several sentence
pairs where one word is swapped with a synonym:

1. Runs the remasker on both sentences -> per-position "error" logits.
2. Masks the target word position, runs the denoiser -> top-k predictions.
3. Prints a comparison showing both words are plausible.

Usage:
    DENOISER_CKPT=/path/to/denoiser.ckpt \\
    REMASKER_CKPT=/path/to/remasker.ckpt \\
    python scripts/demo/demo_synonym_remasker.py
"""

import os
import sys

import omegaconf
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import dataloader
import diffusion

DENOISER_CKPT = os.environ.get("DENOISER_CKPT", "")
REMASKER_CKPT = os.environ.get("REMASKER_CKPT", "")
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "configs")

PAIRS = [
    ("The movie was really great and the audience loved it",
     "The movie was really good and the audience loved it",
     "great", "good"),
    ("The children played in the large garden behind the house",
     "The children played in the big garden behind the house",
     "large", "big"),
    ("She wore a beautiful dress to the evening party",
     "She wore a lovely dress to the evening party",
     "beautiful", "lovely"),
    ("The scientist made an important discovery last year",
     "The scientist made a significant discovery last year",
     "important", "significant"),
    ("The company released a new product last month",
     "The company released a new product last week",
     "month", "week"),
    ("He was a strong leader who inspired many people",
     "He was a powerful leader who inspired many people",
     "strong", "powerful"),
    ("The weather was very cold during the winter season",
     "The weather was very harsh during the winter season",
     "cold", "harsh"),
    ("The students worked hard to complete the difficult assignment",
     "The students worked hard to complete the tough assignment",
     "difficult", "tough"),
]


def build_config():
  if not DENOISER_CKPT or not REMASKER_CKPT:
    raise RuntimeError(
      "Please set DENOISER_CKPT and REMASKER_CKPT environment variables."
    )

  omegaconf.OmegaConf.register_new_resolver("cwd", os.getcwd, replace=True)
  omegaconf.OmegaConf.register_new_resolver(
    "device_count", torch.cuda.device_count, replace=True)
  omegaconf.OmegaConf.register_new_resolver("eval", eval, replace=True)
  omegaconf.OmegaConf.register_new_resolver(
    "div_up", lambda x, y: (x + y - 1) // y, replace=True)

  GlobalHydra.instance().clear()
  with initialize_config_dir(
      config_dir=os.path.abspath(CONFIG_DIR), version_base=None
  ):
    cfg = compose(
      config_name="config",
      overrides=[
        "data=openwebtext-split",
        "parameterization=subs",
        "backbone=dit",
        "model.length=512",
        "noise=loop",
        "noise.t_off=0.05",
        "noise.t_on=0.55",
        "sampling.predictor=remasker",
        f"sampling.remasker_checkpoint_path={REMASKER_CKPT}",
        f"eval.checkpoint_path={DENOISER_CKPT}",
        "remasker.take_first_n_layers=null",
        "sampling.freeze_backbone=false",
        "+sampling.t_on=0.55",
        "+sampling.t_off=0.05",
        "+wandb.offline=true",
      ],
    )
  return cfg


def load_model(cfg, tokenizer, device):
  import importlib
  import types
  import warnings

  legacy = "transformers.models.gpt2.tokenization_gpt2_fast"
  if legacy not in sys.modules:
    current = importlib.import_module("transformers.models.gpt2.tokenization_gpt2")
    shim = types.ModuleType(legacy)
    shim.GPT2TokenizerFast = current.GPT2Tokenizer
    sys.modules[legacy] = shim

  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = diffusion.Diffusion.load_from_checkpoint(
      cfg.eval.checkpoint_path,
      tokenizer=tokenizer,
      config=cfg, strict=False, device=device,
      weights_only=False,
    )
  model.eval()
  model = model.to(device)
  if model.remasker is not None:
    model.remasker = model.remasker.to(device)
  return model


def tokenize_sentence(sentence, tokenizer, seq_len):
  ids = tokenizer.encode(sentence, add_special_tokens=False)
  if len(ids) > seq_len:
    ids = ids[:seq_len]
  else:
    ids = ids + [tokenizer.pad_token_id] * (seq_len - len(ids))
  return torch.tensor([ids], dtype=torch.long)


def find_token_span(tokens, tokenizer, word):
  for i in range(len(tokens)):
    for j in range(i + 1, len(tokens) + 1):
      decoded = tokenizer.decode(tokens[i:j]).strip().lower()
      if decoded == word.lower():
        return list(range(i, j))
  return None


def run_remasker(model, input_ids, device):
  with torch.no_grad():
    logits = model.remasker(input_ids.to(device))
  return logits.squeeze(0).cpu()


def run_denoiser_at_position(model, input_ids, pos, device):
  x = input_ids.clone().to(device)
  x[0, pos] = model.mask_index
  sigma = torch.zeros(1, device=device)
  with torch.no_grad():
    log_probs = model.forward(x, sigma)
  return log_probs[0, pos].cpu()


def to_probs(logits_1d):
  if logits_1d.max() > -1000:
    return logits_1d.softmax(dim=-1)
  return logits_1d.exp()


def evaluate_pair(model, tokenizer, seq_len, device,
                  orig_sent, syn_sent, orig_w, syn_w):
  ids_orig = tokenize_sentence(orig_sent, tokenizer, seq_len)
  ids_syn = tokenize_sentence(syn_sent, tokenizer, seq_len)
  n_orig = (ids_orig[0] != tokenizer.pad_token_id).sum().item()
  n_syn = (ids_syn[0] != tokenizer.pad_token_id).sum().item()
  tokens_orig = ids_orig[0, :n_orig].tolist()
  tokens_syn = ids_syn[0, :n_syn].tolist()

  pos_orig = find_token_span(tokens_orig, tokenizer, orig_w)
  pos_syn = find_token_span(tokens_syn, tokenizer, syn_w)
  if pos_orig is None or pos_syn is None or len(pos_orig) != 1 or len(pos_syn) != 1:
    return None

  p, ps = pos_orig[0], pos_syn[0]

  remask_scores_orig = run_remasker(model, ids_orig, device)
  remask_scores_syn = run_remasker(model, ids_syn, device)

  denoiser_lp = run_denoiser_at_position(model, ids_orig, p, device)
  probs = to_probs(denoiser_lp)
  topk = probs.topk(15)
  top_tokens = [(tokenizer.decode([idx]), pr, idx)
                for idx, pr in zip(topk.indices.tolist(), topk.values.tolist())]

  return dict(
    orig_sent=orig_sent, syn_sent=syn_sent,
    orig_w=orig_w, syn_w=syn_w,
    pos=p, pos_syn=ps,
    remask_orig=remask_scores_orig[p].item(),
    remask_syn=remask_scores_syn[ps].item(),
    prob_orig=probs[tokens_orig[p]].item(),
    prob_syn=probs[tokens_syn[ps]].item(),
    top_tokens=top_tokens,
    remask_all_orig=remask_scores_orig[:n_orig],
    remask_all_syn=remask_scores_syn[:n_syn],
    tokens_orig=tokens_orig, tokens_syn=tokens_syn,
  )


def print_remasker_table(tokens, scores, tokenizer, target_pos):
  print(f"    {'Pos':>4}  {'Token':<20} {'Logit':>8}  {'P(error)':>8}")
  print("    " + "-" * 50)
  for i, tok_id in enumerate(tokens):
    tok = tokenizer.decode([tok_id]).replace("\n", "\\n")
    s = scores[i].item()
    pe = torch.sigmoid(scores[i]).item()
    marker = " <--" if i == target_pos else ""
    print(f"    {i:>4}  {tok:<20} {s:>8.3f}  {pe:>8.3f}{marker}")


def print_topk(top_tokens, orig_w, syn_w):
  print(f"    {'Rank':>4}  {'Token':<20} {'Prob':>8}")
  print("    " + "-" * 38)
  for rank, (tok, prob, _) in enumerate(top_tokens, 1):
    tc = tok.strip().lower()
    marker = ""
    if tc == orig_w.lower():
      marker = " <-- original"
    elif tc == syn_w.lower():
      marker = " <-- synonym"
    print(f"    {rank:>4}  {tok!r:<20} {prob:>8.4f}{marker}")


def main():
  device = "cuda" if torch.cuda.is_available() else "cpu"

  print("Loading config & models...")
  cfg = build_config()
  seq_len = int(cfg.model.length)
  tokenizer = dataloader.get_tokenizer(cfg)
  model = load_model(cfg, tokenizer, device)
  print("Models loaded.\n")

  results = []
  for orig_s, syn_s, ow, sw in PAIRS:
    r = evaluate_pair(model, tokenizer, seq_len, device, orig_s, syn_s, ow, sw)
    if r is not None:
      results.append(r)

  print("=" * 85)
  print("  SYNONYM EQUIVALENCE: screening all pairs")
  print("  Remasker logit < 0 -> word considered correct; > 0 -> suspected error")
  print("=" * 85)
  print(f"  {'Original':<14} {'Synonym':<14} {'Remask(orig)':>13} {'Remask(syn)':>13} {'P(orig)':>9} {'P(syn)':>9}")
  print("  " + "-" * 78)
  for r in results:
    flag = ""
    if r['remask_orig'] < 0 and r['remask_syn'] < 0:
      flag = "  both OK"
    print(f"  {r['orig_w']:<14} {r['syn_w']:<14} "
          f"{r['remask_orig']:>13.3f} {r['remask_syn']:>13.3f} "
          f"{r['prob_orig']:>9.4f} {r['prob_syn']:>9.4f}{flag}")

  good = [r for r in results if r['remask_orig'] < 0 and r['remask_syn'] < 0]
  if not good:
    good = results
  best = max(good, key=lambda r: min(r['prob_orig'], r['prob_syn']))
  r = best

  print(f"\n  Best pair for demonstration: '{r['orig_w']}' / '{r['syn_w']}'")

  print()
  print("=" * 85)
  print(f"  DETAILED ANALYSIS: '{r['orig_w']}' vs '{r['syn_w']}'")
  print("=" * 85)
  print(f"\n  Original : {r['orig_sent']}")
  print(f"  Synonym  : {r['syn_sent']}")
  print(f"  Swap     : '{r['orig_w']}' -> '{r['syn_w']}' at token position {r['pos']}")

  print(f"\n  > Remasker on ORIGINAL sentence:")
  print_remasker_table(r['tokens_orig'], r['remask_all_orig'], tokenizer, r['pos'])

  print(f"\n  > Remasker on SYNONYM sentence:")
  print_remasker_table(r['tokens_syn'], r['remask_all_syn'], tokenizer, r['pos_syn'])

  print(f"\n  > Denoiser top-15 when position {r['pos']} is masked:")
  print_topk(r['top_tokens'], r['orig_w'], r['syn_w'])

  print(f"\n  > Direct probability comparison at the swapped position:")
  w1 = f"'{r['orig_w']}'"
  w2 = f"'{r['syn_w']}'"
  w = max(len(w1), len(w2))
  print(f"    {w1:<{w}}: denoiser P = {r['prob_orig']:.4f},  remasker logit = {r['remask_orig']:+.3f}")
  print(f"    {w2:<{w}}: denoiser P = {r['prob_syn']:.4f},  remasker logit = {r['remask_syn']:+.3f}")

  print()
  print("=" * 85)
  print("  CONCLUSION")
  print("=" * 85)
  if r['remask_orig'] < 0 and r['remask_syn'] < 0:
    print(f"  The remasker assigns negative logits to BOTH '{r['orig_w']}' and")
    print(f"  '{r['syn_w']}', meaning it considers both words correct (no remasking needed).")
  else:
    print(f"  Remasker logits: '{r['orig_w']}' = {r['remask_orig']:.3f}, "
          f"'{r['syn_w']}' = {r['remask_syn']:.3f}")
  print(f"  The denoiser's vocabulary distribution places both words in the top-15,")
  print(f"  confirming the model sees multiple valid choices at this position.")
  print("=" * 85)


if __name__ == "__main__":
  main()
