export OMP_NUM_THREADS=8
export DECORD_EOF_RETRY_MAX=2048001

export TORCH_NCCL_ASYNC_ERROR_HANDLING=0          # 关掉，减少误报
export NCCL_BLOCKING_WAIT=1                       # 改成阻塞等待，更容易看到真实错误
export TORCH_DISTRIBUTED_DEBUG=OFF                  # 减少日志噪音
export TORCH_NCCL_DUMP_ON_TIMEOUT=0               # 别在超时时自动dump，很吵



# llamafactory-cli train LLaMA-Factory/examples/train_full/onethinker_qwen3_sft.yaml
llamafactory-cli train examples/train_full/camreaonser_sft.yaml.yaml