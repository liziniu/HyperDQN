#!/bin/bash
set -e
set -x

script="experiments.deepsea.run_deepsea"
env="deepsea"
seed=2021
size=20
z_size=2
train_iter=20
noise=0.0
l2_norm=0.0
target_update_freq=4
batch_size=128
learning_start=0
eps_greedy=0
double_q=0
discrete_support=0
init='tf'
# prior
prior_mean=0.0
prior_std=1.0
prior_scale=10.0
posterior_scale=1.0
auto_prior=1
# interaction
epoch=200
step_per_epoch=$(( 10*size ))
step_per_collect=1
update_per_step=1
compute_rank_interval=0

export CUDA_VISIBLE_DEVICES=0


if [ "$(uname)" == "Darwin" ]; then
  python -m "$script" \
    --env $env \
    --seed $seed \
    --size $size \
    --z-size $z_size \
    --num-train-iter $train_iter \
    --noise-scale $noise \
    --l2-norm $l2_norm \
    --epoch $epoch \
    --step-per-epoch $step_per_epoch \
    --step-per-collect $step_per_collect \
    --update-per-step $update_per_step \
    --prior-scale $prior_scale \
    --prior-mean $prior_mean \
    --prior-std $prior_std \
    --posterior-scale $posterior_scale \
    --auto-prior $auto_prior \
    --target-update-freq $target_update_freq \
    --batch-size $batch_size \
    --learning-start $learning_start \
    --compute-rank-interval $compute_rank_interval \
    --eps-greedy $eps_greedy \
    --double-q $double_q \
    --discrete-support $discrete_support \
    --init $init
elif [ "$(uname)" == "Linux" ]; then
  for seed in 2021 2022 2023 2024 2025
  do
      python -m "$script" \
        --seed $seed \
        --env $env \
        --size $size \
        --z-size $z_size \
        --num-train-iter $train_iter \
        --noise-scale $noise \
        --l2-norm $l2_norm \
        --epoch $epoch \
        --step-per-epoch $step_per_epoch \
        --step-per-collect $step_per_collect \
        --update-per-step $update_per_step \
        --prior-scale $prior_scale \
        --prior-mean $prior_mean \
        --prior-std $prior_std \
        --posterior-scale $posterior_scale \
        --auto-prior $auto_prior \
        --target-update-freq $target_update_freq \
        --batch-size $batch_size \
        --learning-start $learning_start \
        --compute-rank-interval $compute_rank_interval \
        --eps-greedy $eps_greedy \
        --double-q $double_q \
        --discrete-support $discrete_support \
        --init $init
  done
  wait
fi
