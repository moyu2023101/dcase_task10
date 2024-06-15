#!/bin/bash

# 激活 Poetry 虚拟环境
# source $(poetry env info --path)/bin/activate
conda activate traffic

# 获取传递的第一个参数作为 site 的值
SITE=$1

# 获取传递的第二个参数作为 GPU 的值
GPU=$2
echo "End of script for $SITE :$GPU."
# 检查是否提供了参数
if [ -z "$SITE" ] || [ -z "$GPU" ]; then
  echo "Usage: $0 <site> <gpu>"
  exit 1
fi

# 设置要使用的 GPU
export CUDA_VISIBLE_DEVICES=$GPU

set -e
# 运行 Python 文件
poetry run python -m atsc.counting.training site=$SITE training.learning_rate=0.001 training/train_dataset=sim training/val_dataset=sim training.tags=["train-sim","val-sim"] training.alias=pretrain

poetry run python -m atsc.counting.training site=$SITE training.learning_rate=0.0001 training/train_dataset=real training/val_dataset=real training.tags=["train-real","val-real","finetune"] training.pretrained_model=pretrain training.alias=finetune
poetry run python -m atsc.counting.inference site=$SITE inference.alias=finetune
poetry run python -m atsc.counting.evaluation site=$SITE inference.alias=finetune

echo "End of script for $SITE."