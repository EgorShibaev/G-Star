#!/usr/bin/env bash
# Sample from a trained MDLM denoiser using the guided star-shaped sampler.
# This runs the remasker inside the [0, t_on] interval and plain DDPM elsewhere.
#
# Required environment variables:
#   DENOISER_CKPT    path to the denoiser checkpoint
#   REMASKER_CKPT    path to the remasker checkpoint (trained with remasker_train.py)
#
# Optional: SEQLEN, STEPS, T_ON, T_OFF, NUCLEUS_P, REMASKER_TEMP.

set -euo pipefail

: "${DENOISER_CKPT:?set DENOISER_CKPT to the denoiser checkpoint path}"
: "${REMASKER_CKPT:?set REMASKER_CKPT to the remasker checkpoint path}"
: "${CUDA_VISIBLE_DEVICES:=0}"
: "${SEQLEN:=128}"
: "${STEPS:=128}"
: "${T_ON:=0.3}"
: "${T_OFF:=0.0}"
: "${NUCLEUS_P:=0.9}"
: "${REMASKER_TEMP:=1.0}"

RUN_NAME=${RUN_NAME:-sample-star-shape-seqlen${SEQLEN}-$(date +%Y%m%d_%H%M%S)}

WANDB_API_KEY=${WANDB_API_KEY:-} \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python main.py \
  --multirun \
  'hydra.sweep.dir=multirun_star_shape_seqlen'${SEQLEN}'/${now:%Y-%m-%d}/${now:%H-%M-%S}' \
  'hydra.sweep.subdir="${hydra.job.num}_steps=${sampling.steps}_p=${sampling.nucleus_p}_t=${sampling.remasker_temperature}_t_on=${sampling.t_on}_t_off=${sampling.t_off}_noise=${noise.type}"' \
  mode=sample_eval \
  data=openwebtext-split \
  data.wrap=False \
  parameterization=subs \
  backbone=dit \
  model=small \
  model.length=${SEQLEN} \
  seed=12 \
  loader.batch_size=32 \
  loader.eval_batch_size=32 \
  sampling.num_sample_batches=1 \
  sampling.steps=${STEPS} \
  sampling.predictor=star_shape \
  sampling.remdm_mode=null \
  +sampling.t_on=${T_ON} \
  +sampling.t_off=${T_OFF} \
  sampling.eta=0.008 \
  sampling.remasker_temperature=${REMASKER_TEMP} \
  sampling.remasker_t_off=${T_OFF} \
  sampling.remasker_t_on=${T_ON} \
  sampling.remasker_checkpoint_path=${REMASKER_CKPT} \
  sampling.freeze_backbone=false \
  sampling.nucleus_p=${NUCLEUS_P} \
  sampling.save_x0_sample=true \
  noise=loglinear \
  noise.t_off=${T_OFF} \
  noise.t_on=${T_ON} \
  eval.checkpoint_path=${DENOISER_CKPT} \
  eval.max_trajectories_to_save=64 \
  wandb.name=${RUN_NAME}
