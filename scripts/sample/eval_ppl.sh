#!/usr/bin/env bash
# Zero-shot validation perplexity evaluation of an MDLM checkpoint on OpenWebText.
#
# Required environment variables:
#   CHECKPOINT    checkpoint path (local .ckpt or HuggingFace repo id, e.g. kuleshov-group/mdlm-owt).

set -euo pipefail

: "${CHECKPOINT:?set CHECKPOINT to a checkpoint path or HF repo id}"
: "${CUDA_VISIBLE_DEVICES:=0}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py \
  mode=ppl_eval \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  data=openwebtext-split \
  parameterization=subs \
  backbone=hf_dit \
  model.length=1024 \
  eval.checkpoint_path=${CHECKPOINT}
