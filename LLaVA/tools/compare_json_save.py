import json
import os
from tqdm import tqdm
import copy
import random

def compare_json_files(file1_path, file2_path, sample_size=100):
    """
    Compare two JSON files and find entries that are in file2 but not in file1, ignoring the "unique_id" field.
    Returns random samples of identical entries and entries unique to file2.
    """
    # Check if files exist
    if not os.path.exists(file1_path):
        print(f"File not found: {file1_path}")
        return
    if not os.path.exists(file2_path):
        print(f"File not found: {file2_path}")
        return
    
    # Load JSON files
    print(f"Loading {file1_path}...")
    with open(file1_path, 'r', encoding='utf-8') as f1:
        data1 = json.load(f1)
    
    print(f"Loading {file2_path}...")
    with open(file2_path, 'r', encoding='utf-8') as f2:
        data2 = json.load(f2)
    
    # Check if data is a list
    if not isinstance(data1, list) or not isinstance(data2, list):
        print("Error: JSON files should contain lists")
        return
    
    print(f"File 1 has {len(data1)} entries")
    print(f"File 2 has {len(data2)} entries")
    
    # Process data1 to remove unique_id and create a set for faster lookup
    data1_set = set()
    for item in tqdm(data1, desc="Processing file 1"):
        # Create a copy and remove unique_id if it exists
        item_copy = copy.deepcopy(item)
        if "unique_id" in item_copy:
            del item_copy["unique_id"]
        
        # Convert to string for comparison
        item_str = json.dumps(item_copy, sort_keys=True)
        data1_set.add(item_str)
    
    # Find identical entries and entries unique to file2
    identical_entries = []
    file2_unique_entries = []
    
    for item in tqdm(data2, desc="Comparing file 2 with file 1"):
        # Create a copy and remove unique_id if it exists
        item_copy = copy.deepcopy(item)
        if "unique_id" in item_copy:
            del item_copy["unique_id"]
            
        item_str = json.dumps(item_copy, sort_keys=True)
        if item_str in data1_set:
            identical_entries.append(item)
        else:
            file2_unique_entries.append(item)
    
    print(f"Number of identical entries (ignoring unique_id): {len(identical_entries)}")
    print(f"Number of entries unique to file2: {len(file2_unique_entries)}")
    print(f"Percentage of file2 that is identical to file1: {len(identical_entries)/len(data2)*100:.2f}%")
    print(f"Percentage of file2 that is unique: {len(file2_unique_entries)/len(data2)*100:.2f}%")
    
    # Sample random entries
    random.seed(42)  # For reproducibility
    
    # Sample identical entries
    if len(identical_entries) >= sample_size:
        sampled_identical = random.sample(identical_entries, sample_size)
    else:
        sampled_identical = identical_entries
        print(f"Warning: Only {len(identical_entries)} identical entries found, returning all of them")
    
    # Sample file2 unique entries
    if len(file2_unique_entries) >= sample_size:
        sampled_file2_unique = random.sample(file2_unique_entries, sample_size)
    else:
        sampled_file2_unique = file2_unique_entries
        print(f"Warning: Only {len(file2_unique_entries)} unique entries found in file2, returning all of them")
    
    print(f"Sampled {len(sampled_identical)} identical entries")
    print(f"Sampled {len(sampled_file2_unique)} unique entries from file2")
    
    return sampled_identical, sampled_file2_unique

if __name__ == "__main__":
    file1 = "/data2/jkx/LLaVA/output/final_merged_data_v47_top20.json"
    file2 = "/data2/jkx/LLaVA/output/llava_v1_5_mix665k_random_20pct.json"
    
    # Get identical and file2 unique samples
    identical_samples, file2_unique_samples = compare_json_files(file1, file2, sample_size=100)
    
    # Save results to files
    output_dir = "/data2/jkx/LLaVA/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save identical samples
    identical_file = os.path.join(output_dir, "identical_samples_100.json")
    with open(identical_file, 'w', encoding='utf-8') as f:
        json.dump(identical_samples, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(identical_samples)} identical samples to {identical_file}")
    
    # Save file2 unique samples
    file2_unique_file = os.path.join(output_dir, "file2_unique_samples_100.json")
    with open(file2_unique_file, 'w', encoding='utf-8') as f:
        json.dump(file2_unique_samples, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(file2_unique_samples)} file2 unique samples to {file2_unique_file}") 