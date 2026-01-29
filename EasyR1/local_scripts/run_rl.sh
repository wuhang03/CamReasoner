
#!/usr/bin/env bash
set -x

export DECORD_EOF_RETRY_MAX=2048001

project_name='EasyR1-onethinker-rl'
exp_name='camreasoner-rl'

# MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path
MODEL_PATH="./sft_model"
TRAIN_FILE="camerabench_rl.json"
TEST_FILE="camerabench_rl.json"
IMAGE_DIR=./

ROLLOUT_BS=128 ## 128
GLOBAL_BS=32 ## 32
MB_PER_UPDATE=1 # 1
MB_PER_EXP=1 # 1
VALIDATION_BATCH_SIZE=128
TP_SIZE=4 # should be set according to the number of GPUs used
N_GPUS_PER_NODE=4
NNODES=1



python3 -m verl.trainer.main \
    config=examples/config_ema_grpo_64.yaml \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.image_dir="${IMAGE_DIR}" \
    data.rollout_batch_size="${ROLLOUT_BS}" \
    data.val_batch_size="${VALIDATION_BATCH_SIZE}" \
    worker.actor.global_batch_size="${GLOBAL_BS}" \
    worker.actor.micro_batch_size_per_device_for_update="${MB_PER_UPDATE}" \
    worker.actor.micro_batch_size_per_device_for_experience="${MB_PER_EXP}" \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.optim.lr=2e-6 \
    worker.rollout.tensor_parallel_size="${TP_SIZE}" \
    algorithm.filter_low=0.01 \
    algorithm.filter_high=0.99 \
    algorithm.online_filtering=true \
    algorithm.filter_key=accuracy \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.save_checkpoint_path=./checkpoints


