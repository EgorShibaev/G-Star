#!/usr/bin/env bash
# Sample using the pure remasker-guided sampler (no DDPM fallback inside the
# [t_off, t_on] interval). Uses the `loop` noise schedule.
#
# Required environment variables:
#   DENOISER_CKPT    path to the denoiser checkpoint
#   REMASKER_CKPT    path to the remasker checkpoint

set -euo pipefail

: "${DENOISER_CKPT:?set DENOISER_CKPT to the denoiser checkpoint path}"
: "${REMASKER_CKPT:?set REMASKER_CKPT to the remasker checkpoint path}"
: "${CUDA_VISIBLE_DEVICES:=0}"
: "${SEQLEN:=512}"
: "${STEPS:=128}"
: "${T_ON:=0.55}"
: "${T_OFF:=0.05}"
: "${NUCLEUS_P:=0.9}"
: "${REMASKER_TEMP:=1.0}"

RUN_NAME=${RUN_NAME:-sample-remasker-seqlen${SEQLEN}-$(date +%Y%m%d_%H%M%S)}

WANDB_API_KEY=${WANDB_API_KEY:-} \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python main.py \
  mode=sample_eval \
  data=openwebtext-split \
  data.wrap=False \
  parameterization=subs \
  backbone=dit \
  model=small \
  model.length=${SEQLEN} \
  seed=11 \
  loader.batch_size=32 \
  loader.eval_batch_size=32 \
  sampling.num_sample_batches=64 \
  sampling.steps=${STEPS} \
  sampling.predictor=remasker \
  sampling.remdm_mode=null \
  sampling.remasker_temperature=${REMASKER_TEMP} \
  sampling.remasker_t_off=${T_OFF} \
  sampling.remasker_t_on=${T_ON} \
  sampling.remasker_checkpoint_path=${REMASKER_CKPT} \
  sampling.nucleus_p=${NUCLEUS_P} \
  noise=loop \
  noise.t_off=${T_OFF} \
  noise.t_on=${T_ON} \
  eval.checkpoint_path=${DENOISER_CKPT} \
  wandb.name=${RUN_NAME}
