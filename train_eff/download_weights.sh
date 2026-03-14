#!/bin/bash
# 在登录节点下载预训练权重的脚本

set -e  # 任何命令失败都会退出

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo "开始下载预训练权重"
echo -e "${GREEN}========================================${NC}\n"

# 创建缓存目录
CACHE_DIR="${1:-.}/pretrained_weights"
mkdir -p "$CACHE_DIR"

echo -e "${YELLOW}缓存目录: $CACHE_DIR${NC}\n"

# 激活conda环境（如果需要）
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env

# 执行Python下载脚本
python download_weights.py --cache-dir "$CACHE_DIR"

# 检查缓存内容
echo -e "\n${GREEN}缓存目录内容:${NC}"
find "$CACHE_DIR" -type f -name "*.safetensors" | head -10
echo ""

echo -e "${GREEN}下载完成！${NC}"
echo -e "请使用此环境变量运行训练:"
echo -e "${YELLOW}export HF_HOME=$CACHE_DIR${NC}"
echo -e "${YELLOW}python train.py${NC}"
