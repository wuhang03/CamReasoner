#!/usr/bin/env bash

model_paths=(
  "./sft_model"
)

datasets=(
  # "eval_camerabench.json"
  "camerabench_vqa.json"
  # "camerabench_binary.json"
)


DATASET_PREFIX="./"
OUT_ROOT_BASE="./eval_results"


DIR_SUFFIX="binary"

OUT_DIR="${OUT_ROOT_BASE%/}/${DIR_SUFFIX}"

SUFFIX=""

MAX_PIXELS_VIDEO=$((128*32*32))  # = 262144
MAX_FRAMES=64
BATCH_SIZE=128
FPS=1

export CUDA_VISIBLE_DEVICES=0,1,2,3
export DECORD_EOF_RETRY_MAX=2048001


[ -d "$OUT_DIR" ] || mkdir -p "$OUT_DIR"

for model in "${model_paths[@]}"; do
  for ds_name in "${datasets[@]}"; do
    ds_path="${DATASET_PREFIX%/}/${ds_name}"
    python -u eval/eval_bench.py \
      --model_path "$model" \
      --input_json "$ds_path" \
      --batch_size "$BATCH_SIZE" \
      --out_dir "$OUT_DIR" \
      --suffix "qwen" \
      --max_pixels_video "$MAX_PIXELS_VIDEO" \
      --max_frames "$MAX_FRAMES" \
      --fps "$FPS"
  done
done
