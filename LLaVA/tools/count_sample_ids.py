import torch
import os
from pathlib import Path
from tqdm import tqdm

def count_sample_ids_in_attention_info():
    """
    Count the total number of sample_ids across all PT files in the attention_info directory
    """
    attention_info_dir = Path("/data2/jkx/LLaVA/output/attention_info_lora_one_word")
    
    if not attention_info_dir.exists():
        print(f"Directory {attention_info_dir} does not exist!")
        return
    
    total_sample_ids = 0
    file_count = 0
    sample_ids_set = set()  # To track unique sample_ids if needed
    
    # Get all PT files in the directory
    pt_files = list(attention_info_dir.glob("*.pt"))
    pt_files.sort()  # Sort for consistent ordering
    
    print(f"Found {len(pt_files)} PT files in {attention_info_dir}")
    print("=" * 50)
    
    # Use tqdm for progress bar
    for pt_file in tqdm(pt_files, desc="Processing PT files"):
        try:
            # Load the PT file
            attention_data = torch.load(pt_file)
            
            # Count sample_ids in this file
            file_sample_count = 0
            for sample in attention_data:
                if isinstance(sample, dict) and 'sample_id' in sample:
                    sample_ids_set.add(sample['sample_id'])
                    file_sample_count += 1
            
            total_sample_ids += file_sample_count
            file_count += 1
            
            # print(f"✓ {file_sample_count} sample_ids")
            
        except Exception as e:
            print(f"✗ Error processing {pt_file.name}: {e}")
    
    print("=" * 50)
    print(f"Total files processed: {file_count}")
    print(f"Total sample_ids found: {total_sample_ids}")
    print(f"Unique sample_ids: {len(sample_ids_set)}")
    
    return total_sample_ids, len(sample_ids_set)

if __name__ == "__main__":
    count_sample_ids_in_attention_info() 