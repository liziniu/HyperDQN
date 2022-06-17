#!/bin/bash
set -e
set -x

script="experiments.algos.hyper_dqn.run_atari"
env="PongNoFrameskip-v4"
z_size=32
train_iter=10
noise=0.01
l2_norm=0.01
batch_size=32
target_update_freq=10000
learning_start=50000
# prior
prior_scale=0.1
prior_mean=0.0
prior_std=1.0
posterior_scale=0.1
# interaction
epoch=100
step_per_epoch=50000
step_per_collect=4
update_per_step=0.25

export CUDA_VISIBLE_DEVICES=0

python -m "$script" \
  --env $env \
  --z-size $z_size \
  --num-train-iter $train_iter \
  --noise-scale $noise \
  --l2-norm $l2_norm \
  --epoch $epoch \
  --step-per-epoch $step_per_epoch \
  --update-per-step $update_per_step \
  --step-per-collect $step_per_collect \
  --prior-scale $prior_scale \
  --prior-mean $prior_mean \
  --prior-std $prior_std \
  --posterior-scale $posterior_scale \
  --target-update-freq $target_update_freq \
  --learning-start $learning_start
