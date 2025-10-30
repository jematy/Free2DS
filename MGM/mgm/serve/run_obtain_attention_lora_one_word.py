#!/usr/bin/env python3
import os
import glob
import time
import subprocess
import re

# change the following paths to your own #
MODEL_PATH      = "/data2/jkx/MGM/work_dirs/MGM/MGM-7B"
OUTPUT_PATH     = "/data2/jkx/MGM/output/attention_info_lora_one_word"
IMAGE_FOLDER    = "/data2/jkx/LLaVA/playground/data"
CHUNK_JSON_DIR  = "/data2/jkx/MGM/output/chunk_json"
LOG_DIR         = "/data2/jkx/MGM/output/attention_logs_lora_one_word"  

# Free memory threshold: when GPU memory usage is below this value (MB), it is considered "free" #
FREE_MEM_THRESHOLD = 3072

EXCLUDE_GPUS = {0}

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

def check_output_exists(json_path, output_dir):
    json_basename = os.path.basename(json_path)
    chunk_name = os.path.splitext(json_basename)[0]
    
    output_file = os.path.join(output_dir, f"all_attention_data_{chunk_name}.pt")
    return os.path.exists(output_file)

def main():
    json_files = glob.glob(os.path.join(CHUNK_JSON_DIR, "*.json"))
    json_files.sort(key=extract_number)
    
    if not json_files:
        print("No .json files found in the directory.")
        return
    
    os.makedirs(LOG_DIR, exist_ok=True)
    
    pending_files = []
    skipped_files = []
    
    for json_path in json_files:
        if check_output_exists(json_path, OUTPUT_PATH):
            json_basename = os.path.basename(json_path)
            skipped_files.append(json_basename)
        else:
            pending_files.append(json_path)
    
    if skipped_files:
        print(f"Skipping {len(skipped_files)} already processed files:")
        for f in skipped_files:
            print(f"   - {f}")
        print()
    
    if not pending_files:
        print("All files have been processed, no need to run again.")
        return
    
    print(f"Need to process {len(pending_files)} files:")
    for f in pending_files:
        print(f"   - {os.path.basename(f)}")
    print()

    processes = []
    for json_path in pending_files:
        # Wait for available GPU
        while True:
            gpu_id = find_free_gpu()
            if gpu_id is not None:
                json_basename = os.path.basename(json_path)
                log_file = os.path.join(LOG_DIR, f"{os.path.splitext(json_basename)[0]}.log")
                
                print(f"▶️  Starting on GPU {gpu_id}: {json_basename}, logs saved to {log_file}")
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

                cmd = [
                    "python", "mgm/serve/obtain_attention_one_word.py",
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
                # Use \r to update waiting information on the same line
                wait_time = 0
                while True:
                    gpu_id = find_free_gpu()
                    if gpu_id is not None:
                        break
                    
                    print(f"\rWaiting for free GPU... {wait_time} seconds", end="", flush=True)
                    time.sleep(30)
                    wait_time += 30
                
                # Wait for newline after completion
                print()
    # Wait for all subprocesses to complete
    for p in processes:
        p.wait()

    print("All tasks completed.")

if __name__ == "__main__":
    main()
