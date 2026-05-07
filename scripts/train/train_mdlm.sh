#!/usr/bin/env bash
# Train an MDLM denoiser on OpenWebText.
#
# Required environment variables:
#   WANDB_API_KEY   your Weights & Biases API key
#   CKPT_DIR        root directory for checkpoints
#
# Optional environment variables:
#   CUDA_VISIBLE_DEVICES   GPU devices to use (default: 0,1)
#   RESUME_CKPT            path to a checkpoint to resume from

set -euo pipefail

: "${WANDB_API_KEY:?set WANDB_API_KEY}"
: "${CKPT_DIR:=./checkpoints}"
: "${CUDA_VISIBLE_DEVICES:=0,1}"

RUN_NAME=${RUN_NAME:-mdlm-owt-seqlen512-$(date +%Y%m%d_%H%M%S)}

WANDB_API_KEY=${WANDB_API_KEY} \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python main.py \
  model=small \
  data=openwebtext-split \
  data.wrap=False \
  parameterization=subs \
  model.length=512 \
  wandb.name=${RUN_NAME} \
  eval.compute_generative_perplexity=True \
  sampling.num_sample_batches=4 \
  sampling.steps=128 \
  checkpointing.resume_from_ckpt=${RESUME_CKPT:+true} \
  ${RESUME_CKPT:+checkpointing.resume_ckpt_path=${RESUME_CKPT}} \
  checkpointing.save_dir=${CKPT_DIR}/${RUN_NAME} \
  loader.global_batch_size=512 \
  loader.batch_size=128 \
  loader.eval_batch_size=128 \
  trainer.accumulate_grad_batches=2 \
  trainer.val_check_interval=5000
