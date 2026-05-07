#!/usr/bin/env bash
# Train the remasker with the `random_corruption` strategy: labels are generated
# by randomly corrupting a fraction of x0 tokens; the diffusion pipeline is NOT
# used. The corruption mask is the target.
#
# Required environment variables:
#   WANDB_API_KEY    your Weights & Biases API key
#   DENOISER_CKPT    path to a trained MDLM denoiser checkpoint
#   CKPT_DIR         root directory where checkpoints will be saved

set -euo pipefail

: "${WANDB_API_KEY:?set WANDB_API_KEY}"
: "${DENOISER_CKPT:?set DENOISER_CKPT to the denoiser checkpoint path}"
: "${CKPT_DIR:=./checkpoints}"
: "${CUDA_VISIBLE_DEVICES:=0}"

RUN_NAME=${RUN_NAME:-remasker-random-corruption-p0.1-seqlen512-$(date +%Y%m%d_%H%M%S)}

export TOKENIZERS_PARALLELISM=false

WANDB_API_KEY=${WANDB_API_KEY} \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python remasker_train.py \
  eval.checkpoint_path=${DENOISER_CKPT} \
  checkpointing.save_dir=${CKPT_DIR}/${RUN_NAME} \
  checkpointing.resume_from_ckpt=false \
  wandb.name=${RUN_NAME} \
  +trainer.max_epochs=10 \
  model=small \
  model.length=512 \
  loader.batch_size=64 \
  loader.eval_batch_size=64 \
  trainer.accumulate_grad_batches=8 \
  loader.global_batch_size=512 \
  trainer.precision=bf16 \
  trainer.val_check_interval=1000 \
  data=openwebtext-split \
  data.wrap=false \
  optim.lr=1e-4 \
  callbacks.checkpoint_monitor.monitor=val/loss \
  sampling.freeze_backbone=false \
  remasker.take_first_n_layers=null \
  remasker.training_strategy=random_corruption \
  remasker.corruption_ratio=0.1 \
  noise=loglinear \
  training.remasker_reweighting=false
