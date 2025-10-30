#!/bin/bash

# 获取 GPU 列表
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# 计算 GPU 数量
CHUNKS=${#GPULIST[@]}

# 定义模型路径和数据集路径,改CKPT也许就可以是不同的版本
#CKPT="llava-v1.5-13b"
CKPT="llava-v1.5-full"



# # 遍历所有 GPU 进行计算
for IDX in $(seq 0 $((CHUNKS-1))); do


    # 分配 GPU 并执行模型评估
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path /data2/cwx/icons/checkpoints/llava_full_merged \
        --question-file /data2/cwx/icons/eval/chartqa/chartqa_questions.jsonl \
        --image-folder /data2/cwx/icons/eval/chartqa \
        --answers-file /data2/cwx/icons/eval/chartqa/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &  # 后台运行
done

wait

# 合并所有的输出文件
output_file=/data2/cwx/icons/eval/chartqa/$CKPT/merge.jsonl

# 清空合并输出文件（如果已存在）
> "$output_file"

# 合并所有 GPU 结果
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /data2/cwx/icons/eval/chartqa/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# # 转换结果为提交格式
# CUDA_VISIBLE_DEVICES="1,2" python /home/cwx/LLaVA/scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT
python ./evaluate_realworldqa.py --pred_file /data2/cwx/icons/eval/chartqa/$CKPT/merge.jsonl --answer_file /data2/cwx/icons/eval/chartqa/chartqa_answers.jsonl
