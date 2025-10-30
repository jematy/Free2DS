#!/usr/bin/env python3
import os
import glob
import time
import subprocess
import re

MODEL_PATH      = "/data2/jkx/LLaVA/checkpoints/llava-v1.5-7b"
OUTPUT_PATH     = "/data2/jkx/LLaVA/output/attention_info_one_word"
IMAGE_FOLDER    = "/data2/jkx/LLaVA/playground/data"
CHUNK_JSON_DIR  = "/data2/jkx/LLaVA/output/chunk_json"
LOG_DIR         = "/data2/jkx/LLaVA/output/attention_logs_one_word"  

FREE_MEM_THRESHOLD = 5000

EXCLUDE_GPUS = {}

def find_free_gpu(threshold=FREE_MEM_THRESHOLD, exclude=EXCLUDE_GPUS):
    exclude = set() if exclude is None else set(exclude)
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print("Error finding free GPU:", res.stderr.strip())
        return None

    usages = [int(x) for x in res.stdout.strip().splitlines()]
    for idx, used in enumerate(usages):
        if idx in exclude:
            continue
        if used < threshold:
            return idx
    return None

def extract_number(filename):
    match = re.search(r'(\d+)', os.path.basename(filename))
    if match:
        return int(match.group(1))
    return 0

def main():
    json_files = glob.glob(os.path.join(CHUNK_JSON_DIR, "*.json"))
    json_files.sort(key=extract_number)

    start_num = 123
    json_files = [f for f in json_files if extract_number(f) >= start_num]
    
    if not json_files:
        print("No .json files found in the directory.")
        return
    
    os.makedirs(LOG_DIR, exist_ok=True)

    processes = []
    for json_path in json_files:
        while True:
            gpu_id = find_free_gpu()
            if gpu_id is not None:
                json_basename = os.path.basename(json_path)
                log_file = os.path.join(LOG_DIR, f"{os.path.splitext(json_basename)[0]}.log")
                
                print(f"Starting on GPU {gpu_id}: {json_basename}, logs saved to {log_file}")
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

                cmd = [
                    "python", "obtain_attention_one_word.py",
                    "--model_path",    MODEL_PATH,
                    "--output_path",   OUTPUT_PATH,
                    "--image_folder",  IMAGE_FOLDER,
                    "--data_path",     json_path
                ]
                
                with open(log_file, 'w') as f:
                    p = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
                    processes.append(p)
                
                time.sleep(30)
                break
            else:
                wait_time = 0
                while True:
                    gpu_id = find_free_gpu()
                    if gpu_id is not None:
                        break
                    
                    print(f"\rWaiting for free GPU... {wait_time} seconds", end="", flush=True)
                    time.sleep(30)
                    wait_time += 10
                
                print()

    for p in processes:
        p.wait()

    print("All tasks completed.")

if __name__ == "__main__":
    main()
