#!/usr/bin/env bash
# Sample using the Path-Planning (P2) baseline (Peng et al., 2025).
#
# Required environment variables:
#   DENOISER_CKPT    path to the denoiser checkpoint

set -euo pipefail

: "${DENOISER_CKPT:?set DENOISER_CKPT to the denoiser checkpoint path}"
: "${CUDA_VISIBLE_DEVICES:=0}"
: "${SEQLEN:=512}"
: "${STEPS:=128}"
: "${T_ON:=1.0}"
: "${T_OFF:=0.0}"
: "${NUCLEUS_P:=0.9}"
: "${P2_TEMP:=1.0}"

RUN_NAME=${RUN_NAME:-sample-p2-seqlen${SEQLEN}-$(date +%Y%m%d_%H%M%S)}

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
  loader.batch_size=64 \
  loader.eval_batch_size=64 \
  sampling.num_sample_batches=64 \
  sampling.steps=${STEPS} \
  sampling.predictor=p2 \
  sampling.nucleus_p=${NUCLEUS_P} \
  p2.t_off=${T_OFF} \
  p2.t_on=${T_ON} \
  p2.temperature=${P2_TEMP} \
  p2.nucleus_p=${NUCLEUS_P} \
  p2.use_argmax=false \
  p2.confidence_type=prob \
  noise=loglinear \
  eval.checkpoint_path=${DENOISER_CKPT} \
  wandb.name=${RUN_NAME}
