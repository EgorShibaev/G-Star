#!/usr/bin/env bash
# Sample using ReMDM (cap mode, DDPM-cache predictor). Baseline from
# "ReMDM: Remasking Diffusion Models" (Wang et al., 2024).
#
# Required environment variables:
#   DENOISER_CKPT    path to the denoiser checkpoint
#
# Optional: SEQLEN, STEPS, ETA, NUCLEUS_P.

set -euo pipefail

: "${DENOISER_CKPT:?set DENOISER_CKPT to the denoiser checkpoint path}"
: "${CUDA_VISIBLE_DEVICES:=0}"
: "${SEQLEN:=512}"
: "${STEPS:=128}"
: "${ETA:=0.008}"
: "${NUCLEUS_P:=0.9}"

RUN_NAME=${RUN_NAME:-sample-remdm-seqlen${SEQLEN}-eta${ETA}-$(date +%Y%m%d_%H%M%S)}

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
  sampling.predictor=ddpm_cache \
  sampling.remdm_mode=cap \
  sampling.eta=${ETA} \
  sampling.nucleus_p=${NUCLEUS_P} \
  noise=loglinear \
  eval.checkpoint_path=${DENOISER_CKPT} \
  wandb.name=${RUN_NAME}
