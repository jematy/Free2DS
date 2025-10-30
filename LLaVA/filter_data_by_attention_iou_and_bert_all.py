#这里筛选只会有<image>
import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
import json
import random
from scipy.special import softmax


def load_attention_data(file_path):
    """Load attention data from a PT file."""
    try:
        data = torch.load(file_path)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def calculate_iou(attention1, attention2, top_p=30):
    """
    Calculate IoU of top p% vision tokens with highest scores.
    
    Args:
        attention1: Normalized attention vector from model 1 [vision_tokens]
        attention2: Normalized attention vector from model 2 [vision_tokens]
        top_p: Percentage of top tokens to consider (default: 30%)
    
    Returns:
        IoU score
    """
    num_tokens = attention1.size(0)
    k = max(1, int(num_tokens * top_p / 100))
    
    # Get indices of top k tokens for each attention
    _, top_indices1 = torch.topk(attention1, k)
    _, top_indices2 = torch.topk(attention2, k)
    
    # Convert to sets for intersection and union
    set1 = set(top_indices1.tolist())
    set2 = set(top_indices2.tolist())
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    iou = intersection / union if union > 0 else 0.0
    return iou


def aggregate_token_attention(attention_data):
    """
    Aggregate attention across tokens for each sample.
    Returns a dict mapping sample_id to aggregated attention vector.
    """
    sample_to_attention = {}
    
    for sample in attention_data:
        sample_id = sample['sample_id']
        rel_attention_maps = sample['relative_attention_maps']  # [tokens, vision_tokens]
        
        # Aggregate across tokens by summing and normalize
        if len(rel_attention_maps) > 0:
            aggregated_attention = rel_attention_maps.sum(dim=0)  # [vision_tokens]
            if aggregated_attention.sum() > 0:
                aggregated_attention = aggregated_attention / aggregated_attention.sum()
            sample_to_attention[sample_id] = aggregated_attention
    
    return sample_to_attention


def process_chunk_pair(model1_file, model2_file):
    """Process a pair of chunk files and calculate IoU for top 30% vision tokens."""
    model1_data = load_attention_data(model1_file)
    model2_data = load_attention_data(model2_file)
    
    if model1_data is None or model2_data is None:
        return None
    
    # Aggregate attention for each sample
    model1_attention = aggregate_token_attention(model1_data)
    model2_attention = aggregate_token_attention(model2_data)
    
    # Find common samples
    common_samples = set(model1_attention.keys()).intersection(set(model2_attention.keys()))
    
    if not common_samples:
        print(f"No common samples found between {model1_file} and {model2_file}")
        return None
    
    # Calculate IoU for each common sample (using top 30%)
    sample_ious = {}
    for sample_id in common_samples:
        attn1 = model1_attention[sample_id]
        attn2 = model2_attention[sample_id]
        iou = calculate_iou(attn1, attn2, top_p=30)
        sample_ious[sample_id] = iou
    
    return sample_ious


def load_json_data(json_file):
    """Load JSON data file."""
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {json_file}: {e}")
        return None


def save_filtered_data(data, selected_ids, output_file):
    """Save filtered data to a new JSON file."""
    filtered_data = [item for item in data if item.get("unique_id") in selected_ids]
    
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    return len(filtered_data)


def load_bert_scores(bert_scores_file):
    """Load BERT similarity scores from a JSON file."""
    try:
        with open(bert_scores_file, 'r') as f:
            bert_scores = json.load(f)
        print(f"Loaded BERT scores for {len(bert_scores)} samples")
        return bert_scores
    except Exception as e:
        print(f"Error loading BERT scores from {bert_scores_file}: {e}")
        return {}


def main():

    
    final_output_path = "/data2/jkx/LLaVA/output/final_merged_data.json"


    parser = argparse.ArgumentParser(description="Filter data based on attention IoU scores and BERT similarity")
    parser.add_argument("--model1_dir", type=str, default="/data2/jkx/LLaVA/output/attention_info_one_word/",
                        help="Directory containing attention files for model 1")
    parser.add_argument("--model2_dir", type=str, default="/data2/jkx/MGM/output/attention_info_one_word/",
                        help="Directory containing attention files for model 2")
    parser.add_argument("--data_file", type=str, default="/data2/jkx/LLaVA/playground/data/llava_v1_5_mix665k_with_unique_id.json",
                        help="Path to the JSON data file to filter")
    parser.add_argument("--output_dir", type=str, default="/data2/jkx/LLaVA/filtered_data_all/",
                        help="Directory to save filtered data")
    parser.add_argument("--bert_scores_file", type=str, default="/data2/jkx/LLaVA/output/conversation_scores.json",
                        help="File containing BERT similarity scores")
    parser.add_argument("--filter_ratio", type=float, default=0.2,
                        help="Percentage of data to keep (0.2 means keep 20%)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Exact number of samples to keep (overrides filter_ratio if specified)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for softmax function")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for combining IoU and BERT scores (0: only IoU, 1: only BERT)")
    parser.add_argument("--combined_output_name", type=str, default="final_merged_data",
                        help="Name for the combined output file (without .json extension)")
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define the final output path using the parameter
    final_output_path = f"/data2/jkx/LLaVA/output/{args.combined_output_name}.json"
    
    # Get all chunk files from both directories
    model1_files = {os.path.basename(f): f for f in 
                    [os.path.join(args.model1_dir, f) for f in os.listdir(args.model1_dir) 
                     if f.endswith('.pt') and os.path.isfile(os.path.join(args.model1_dir, f))]}
    
    model2_files = {os.path.basename(f): f for f in 
                    [os.path.join(args.model2_dir, f) for f in os.listdir(args.model2_dir) 
                     if f.endswith('.pt') and os.path.isfile(os.path.join(args.model2_dir, f))]}
    
    # Find common chunks
    common_chunks = set(model1_files.keys()).intersection(set(model2_files.keys()))
    print(f"Found {len(common_chunks)} common chunk files")
    
    # Collect IoU scores for all samples
    all_sample_ious = {}
    
    for chunk_name in tqdm(sorted(common_chunks), desc="Processing chunks"):
        model1_file = model1_files[chunk_name]
        model2_file = model2_files[chunk_name]
        
        sample_ious = process_chunk_pair(model1_file, model2_file)
        
        if sample_ious:
            all_sample_ious.update(sample_ious)
    
    print(f"Collected IoU scores for {len(all_sample_ious)} samples")
    
    # Load original data to get all sample IDs
    print("Loading data file to get all sample IDs...")
    data = load_json_data(args.data_file)
    if not data:
        print("Error loading data file. Exiting.")
        return
    
    # Extract all unique IDs from the data
    all_sample_ids = [item.get("unique_id") for item in data if "unique_id" in item]
    print(f"Found {len(all_sample_ids)} samples in data file")
    
    # Load BERT similarity scores
    bert_scores = load_bert_scores(args.bert_scores_file)
    
    # Create a list of all sample IDs and their scores, defaulting IoU to 0 for missing samples
    sample_ids = all_sample_ids
    combined_scores = []
    
    # Collect all IoU and BERT scores for normalization
    iou_scores_list = []
    bert_scores_list = []
    
    for sid in sample_ids:
        # Get IoU score or default to 0 if not found
        iou_score = 0.0
        if sid in all_sample_ious:
            iou_score = all_sample_ious[sid]
            # iou_score = 1.0 - all_sample_ious[sid]
        
        # Convert string sample_id to match the format in bert_scores
        bert_score = bert_scores.get(str(sid), 0.0)
        
        iou_scores_list.append(iou_score)
        bert_scores_list.append(bert_score)
    
    # Normalize IOU and BERT scores to [0, 1] range
    iou_scores_array = np.array(iou_scores_list)
    bert_scores_array = np.array(bert_scores_list)
    
    # Avoid division by zero
    iou_min, iou_max = iou_scores_array.min(), iou_scores_array.max()
    bert_min, bert_max = bert_scores_array.min(), bert_scores_array.max()
    
    if iou_max > iou_min:
        normalized_iou_scores = (iou_scores_array - iou_min) / (iou_max - iou_min)
        normalized_iou_scores = 1 - normalized_iou_scores
    else:
        normalized_iou_scores = np.zeros_like(iou_scores_array)
    
    if bert_max > bert_min:
        normalized_bert_scores = (bert_scores_array - bert_min) / (bert_max - bert_min)
        normalized_bert_scores = 1 - normalized_bert_scores
    else:
        normalized_bert_scores = np.zeros_like(bert_scores_array)
    
    # Combine the normalized scores using the alpha parameter (weighted average)
    for i, sid in enumerate(sample_ids):
        combined_score = (1 - args.alpha) * normalized_iou_scores[i] + args.alpha * normalized_bert_scores[i]
        combined_scores.append(combined_score)
    
    combined_scores = np.array(combined_scores)
    
    # Apply softmax to combined scores
    temperature = args.temperature
    softmax_scores = softmax(combined_scores / temperature)
    
    # Calculate number of samples to keep
    if args.num_samples is not None and args.num_samples > 0:
        num_samples_to_keep = min(args.num_samples, len(sample_ids))
        actual_ratio = num_samples_to_keep / len(sample_ids)
        print(f"Selecting exactly {num_samples_to_keep} samples out of {len(sample_ids)} ({actual_ratio * 100:.1f}%)")
    else:
        num_samples_to_keep = int(len(sample_ids) * args.filter_ratio)
        print(f"Selecting {num_samples_to_keep} samples out of {len(sample_ids)} ({args.filter_ratio * 100:.1f}%)")
    
    # Sample based on softmax probabilities
    selected_indices = np.random.choice(
        len(sample_ids), 
        size=num_samples_to_keep, 
        replace=False, 
        p=softmax_scores/softmax_scores.sum()
    )
    
    selected_sample_ids = [sample_ids[i] for i in selected_indices]
    
    # Save the list of selected sample IDs
    with open(os.path.join(args.output_dir, "selected_sample_ids.json"), 'w') as f:
        json.dump(selected_sample_ids, f)
    
    # Save the combined scores for analysis
    scores_info = {}
    for i, sid in enumerate(sample_ids):
        iou = all_sample_ious.get(sid, 0.0)
        bert = bert_scores.get(str(sid), 0.0)
        scores_info[str(sid)] = {
            "iou": float(iou),
            "iou_normalized": float(normalized_iou_scores[i]),
            "bert": float(bert),
            "bert_normalized": float(normalized_bert_scores[i]),
            "combined": float(combined_scores[i]),
            "has_iou_score": sid in all_sample_ious
        }
    
    with open(os.path.join(args.output_dir, "sample_scores.json"), 'w') as f:
        json.dump(scores_info, f, indent=2)
    
    # Process the single JSON data file and filter based on selected sample IDs
    output_path = os.path.join(args.output_dir, "filtered_data.json")
    num_filtered = save_filtered_data(data, selected_sample_ids, output_path)
    print(f"Filtered {num_filtered} samples from {len(data)} total samples")
    
    print(f"Filtered data saved to {output_path}")
    
    # Also save to the output directory with specific name
    
    final_num_filtered = save_filtered_data(data, selected_sample_ids, final_output_path)
    print(f"Final merged data saved to {final_output_path}")
    
    # Count how many selected samples had IoU scores
    selected_with_iou = sum(1 for sid in selected_sample_ids if sid in all_sample_ious)
    
    # Save statistics
    stats = {
        "total_samples": len(sample_ids),
        "selected_samples": len(selected_sample_ids),
        "selected_with_iou": selected_with_iou,
        "selected_without_iou": len(selected_sample_ids) - selected_with_iou,
        "filter_ratio": args.filter_ratio if args.num_samples is None else (num_samples_to_keep / len(sample_ids)),
        "num_samples_requested": args.num_samples,
        "input_file": args.data_file,
        "total_filtered_samples": num_filtered,
        "alpha": args.alpha
    }
    
    with open(os.path.join(args.output_dir, "filter_stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main() 