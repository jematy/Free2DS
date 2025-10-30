#!/bin/bash

MODEL_PATH="/data2/jkx/LLaVA/checkpoints/llava-v1.5-7b-lora-v47-top20-per005-lora"
EVAL_PATH="/data2/jkx/LLaVA/playground/data/eval/ai2d"
CKPT="llava-v1.5-7b-lora-v47-top20-per005-lora"
python -m llava.eval.model_vqa_science \
    --model-path $MODEL_PATH \
    --question-file $EVAL_PATH/realworldqa_dialogue.json \
    --image-folder $EVAL_PATH/\
    --answers-file $EVAL_PATH/answers/${CKPT}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1


python /data2/jkx/LLaVA/scripts/v1_5/eval/evaluate_ai2d.py --pred_file $EVAL_PATH/answers/${CKPT}.jsonl --gt_file /data2/jkx/LLaVA/playground/data/eval/ai2d/realworldqa_answers.jsonl
# CUDA_VISIBLE_DEVICES="1" python llava/eval/eval_science_qa.py \
#     --base-dir $EVAL_PATH \
#     --result-file $EVAL_PATH/answers/${CKPT}.jsonl \
#     --output-file $EVAL_PATH/answers/${CKPT}-output.jsonl \
#     --output-result $EVAL_PATH/answers/${CKPT}-result.json
