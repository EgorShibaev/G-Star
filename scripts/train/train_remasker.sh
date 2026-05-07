#!/usr/bin/env bash
# Train the remasker (learned error predictor g_phi) against a frozen MDLM
# denoiser using the default BCE strategy.
#
# Required environment variables:
#   WANDB_API_KEY        your Weights & Biases API key
#   DENOISER_CKPT        path to a trained MDLM denoiser checkpoint
#   CKPT_DIR             root directory where the remasker's checkpoints will go

set -euo pipefail

: "${WANDB_API_KEY:?set WANDB_API_KEY}"
: "${DENOISER_CKPT:?set DENOISER_CKPT to the denoiser checkpoint path}"
: "${CKPT_DIR:=./checkpoints}"
: "${CUDA_VISIBLE_DEVICES:=0}"

RUN_NAME=${RUN_NAME:-remasker-seqlen512-$(date +%Y%m%d_%H%M%S)}

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
  sampling.t_sampling=uniform \
  sampling.nucleus_p=1.0 \
  sampling.freeze_backbone=false \
  remasker.take_first_n_layers=null \
  remasker.training_strategy=default \
  noise=loglinear \
  training.remasker_reweighting=false
