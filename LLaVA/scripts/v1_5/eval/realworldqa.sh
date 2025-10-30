#!/bin/bash

# 获取 GPU 列表
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# 计算 GPU 数量
CHUNKS=${#GPULIST[@]}

# 定义模型路径和数据集路径,改CKPT也许就可以是不同的版本
#CKPT="llava-v1.5-7b-lora-v47-top20-per005-lora"
CKPT="llava-v1.5-7b-lora-v47-top20-per005-lora"
MODEL_PATH="/data2/jkx/LLaVA/checkpoints/llava-v1.5-7b-lora-v47-top20-per005-lora"


# # 遍历所有 GPU 进行计算
for IDX in $(seq 0 $((CHUNKS-1))); do


    # 分配 GPU 并执行模型评估
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $MODEL_PATH \
        --question-file /data2/jkx/LLaVA/playground/data/eval/realworldqa/realworldqa_vqa_format.jsonl \
        --image-folder /data2/jkx/LLaVA/playground/data/eval/realworldqa/ \
        --answers-file /data2/jkx/LLaVA/playground/data/eval/realworldqa/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &  # 后台运行
done

wait

# 合并所有的输出文件
output_file=/data2/jkx/LLaVA/playground/data/eval/realworldqa/$CKPT/merge.jsonl

# 清空合并输出文件（如果已存在）
> "$output_file"

# 合并所有 GPU 结果
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /data2/jkx/LLaVA/playground/data/eval/realworldqa/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# # 转换结果为提交格式
# CUDA_VISIBLE_DEVICES="1,2" python /home/cwx/LLaVA/scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT
python /data2/jkx/LLaVA/scripts/v1_5/eval/evaluate_realworldqa.py --pred_file /data2/jkx/LLaVA/playground/data/eval/realworldqa/$CKPT/merge.jsonl --answer_file /data2/jkx/LLaVA/playground/data/eval/realworldqa/realworldqa_v2_answers.jsonl
