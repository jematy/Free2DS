"""
    This script is used for getting gradients or representations of a pre-trained model for a given task.
"""
import os
import glob
import datetime
import argparse
from typing import List, Optional

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
from tqdm import tqdm

# Import this directly since icons.utils import was causing errors
from obtain_attention_utils import LazySupervisedDataset  # Direct import instead of * import
from mgm.model.builder import load_pretrained_model
from mgm.mm_utils import get_model_name_from_path
from mgm.constants import IMAGE_TOKEN_INDEX


def save_detailed_model_parameters(model, output_file="detailed_model_parameters.txt"):
    """Save detailed model parameters including actual tensor values to a file."""
    with open(output_file, 'w') as f:
        f.write(f"Detailed Model Parameters Dump - {datetime.datetime.now()}\n")
        f.write("=" * 80 + "\n\n")
        
        for name, param in model.named_parameters():
            f.write(f"{name}: Parameter containing:\n")
            tensor_str = str(param.data)
            f.write(tensor_str)
            f.write(f"\nShape: {tuple(param.shape)}")
            f.write(f"\nRequires grad: {param.requires_grad}")
            f.write(f"\nDevice: {param.device}")
            f.write(f"\nDtype: {param.dtype}\n")
            f.write("\n" + "-" * 80 + "\n")


def load_raw_dataset(data_args, tokenizer, train_files):
    print(f"Train files received: {train_files}")
    
    if train_files is None:
        raise ValueError("train_files cannot be None. Please provide a valid data path.")
    
    dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=train_files,
        data_args=data_args
    )
    
    return dataset


def get_dataset(data_args, files: List[str], tokenizer, max_seq_length):
    raw_datasets = load_raw_dataset(data_args, tokenizer, files)
    
    class TruncatedDataset:
        def __init__(self, dataset, max_length):
            self.dataset = dataset
            self.max_length = max_length
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            item = self.dataset[idx]
            image = item.get("image", None)
            unique_id = item.get("unique_id", None)
            
            if len(item["input_ids"]) > self.max_length:
                item["input_ids"] = item["input_ids"][:self.max_length]
                if "labels" in item:
                    item["labels"] = item["labels"][:self.max_length]
            
            if not isinstance(item["input_ids"], torch.Tensor):
                item["input_ids"] = torch.tensor(item["input_ids"])
            if "labels" in item and not isinstance(item["labels"], torch.Tensor):
                item["labels"] = torch.tensor(item["labels"])
            
            if image is not None:
                item["image"] = image
            
            # Preserve unique_id if it exists
            if unique_id is not None:
                item["unique_id"] = unique_id
            
            return item
    
    return TruncatedDataset(raw_datasets, max_seq_length)


def get_dataloader(dataset, tokenizer, batch_size=1):
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest")
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
    print(f"There are {len(dataset)} examples in the dataset")
    return dataloader


def initialize_model(args):
    print("Loading model from:", args.model_path)
    
    # Simplified model loading similar to inference.py
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=get_model_name_from_path(args.model_path),
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded successfully and moved to {device}")

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        print("Token embeddings resized")

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    return tokenizer, model, image_processor


def check_data(args, dataloader, model, tokenizer):
    """Check data."""
    
    print("=" * 60)
    print("check data")
    print("=" * 60)
    
    try:
        first_batch = next(iter(dataloader))
        print(f"Batch type: {type(first_batch)}")
        print(f"Batch keys: {list(first_batch.keys())}")
        
        for key, value in first_batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: tensor shape {value.shape}, dtype {value.dtype}")
                if key in ['input_ids', 'labels']:
                    print(f"    {key} first 10 values: {value[0][:10] if value.dim() > 1 else value[:10]}")
                    print(f"    {key} length: {value.shape[-1]}")
            else:
                print(f"  {key}: {type(value)} - {value}")
                
        print(first_batch['unique_id'][0])
        print("Labels shape:", first_batch['labels'].shape)
        print("Input_ids shape:", first_batch['input_ids'].shape)
        # print("Labels:", first_batch['labels'])
        if 'unique_id' in first_batch:
            print("Unique_id:", first_batch['unique_id'])
        else:
            print("Unique_id: Not present in this batch")
        # labels_sample = first_batch['labels'][0]
        # input_ids_sample = first_batch['input_ids'][0]
        
        # # neg_indices = torch.where(labels_sample < 0)[0]
        # neg_indices = []
        # for idx, label in enumerate(labels_sample):
        #     if label == -100 != first_batch['input_ids'][0][idx]:
        #         neg_indices.append(idx)
        #     else:
        #         print(f"Break at idx={idx}, label={label}")
        #         break
        # print("Negative label indices:", neg_indices)
        
        # if len(neg_indices) > 0:
        #     # get the corresponding input_ids
        #     target_tokens = input_ids_sample[neg_indices]
        #     print("Target tokens:", target_tokens)
        #     decoded_text = tokenizer.decode(target_tokens[40:54])
        #     print("Decoded target text:", decoded_text)
        # else:
        #     print("No positive labels found")

        if 'image' in first_batch:
            image_tensor = first_batch['image']
            print(f"  Image data: shape {image_tensor.shape}, dtype {image_tensor.dtype}")
            print(f"  Image value range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
        
        print(f"\nData loader length: {len(dataloader)}")
        print(f"Batch size: {first_batch['input_ids'].shape[0] if 'input_ids' in first_batch else 'Unknown'}")
        
    except Exception as e:
        print(f"Error checking data structure: {e}")
        import traceback
        traceback.print_exc()
    

def process_sample_attention(model, tokenizer, sample_input_ids, sample_image, sample_image_aux, sample_id, vision_token_start, vision_token_end, output_path):
    """Process a single sample and collect attention between generated tokens and vision tokens."""
    with torch.inference_mode():
        try:
            
            output_ids = model.generate(
                sample_input_ids,
                images=sample_image,
                images_aux=sample_image_aux,
                do_sample=False,
                temperature=0,
                max_new_tokens=100,
                bos_token_id=tokenizer.bos_token_id,  # Begin of sequence token
                eos_token_id=tokenizer.eos_token_id,  # End of sequence token
                pad_token_id=tokenizer.pad_token_id,  # Pad token
                use_cache=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )
            
            attention_data = output_ids["attentions"]
            
            all_token_ids = []
            all_attention_maps = []
            all_relative_attention_maps = []
            
            last_token_idx = len(attention_data) - 1
            if last_token_idx < 0:
                print(f"Warning: No tokens generated for sample {sample_id}")
                return None
                
            last_token_attention_map = torch.zeros((vision_token_end - vision_token_start,), device=model.device)
            
            last_attn_step = attention_data[last_token_idx]
            for layer_idx, layer_attn in enumerate(last_attn_step):
                layer_attn_aggregated = layer_attn.mean(dim=1)  # [batch_size, heads, seq_len, seq_len]
                layer_attn_score = layer_attn_aggregated[0, -1, vision_token_start:vision_token_end]
                last_token_attention_map += layer_attn_score
                
            last_token_attention_map /= len(last_attn_step)
            
            epsilon = 1e-10
            last_token_attention_map = last_token_attention_map + epsilon
            
            for t, attn_step in enumerate(attention_data):
                if t >= len(attention_data) - 1:
                    break
                
                token_attention_map = torch.zeros((vision_token_end - vision_token_start,), device=model.device)
                
                for layer_idx, layer_attn in enumerate(attn_step):
                    layer_attn_aggregated = layer_attn.mean(dim=1)  # [batch_size, heads, seq_len, seq_len]
                    layer_attn_score = layer_attn_aggregated[0, -1, vision_token_start:vision_token_end]
                    token_attention_map += layer_attn_score
                
                token_attention_map /= len(attn_step)
                
                relative_attention_map = token_attention_map / last_token_attention_map
                
                if relative_attention_map.sum() > 0:
                    relative_attention_map = relative_attention_map / relative_attention_map.sum()
                
                # get the current token ID
                current_token_id = output_ids["sequences"][0][t].item()
                
                all_token_ids.append(current_token_id)
                all_attention_maps.append(token_attention_map)
                all_relative_attention_maps.append(relative_attention_map)
            
            # stack all token attention maps into a single tensor: [tokens, vision_tokens]
            if all_attention_maps:
                attention_tensor = torch.stack(all_attention_maps)
                relative_attention_tensor = torch.stack(all_relative_attention_maps)
            else:
                attention_tensor = torch.tensor([])
                relative_attention_tensor = torch.tensor([])
            
            token_ids_tensor = torch.tensor(all_token_ids, dtype=torch.long)
            
            
            attention_data = {
                'sample_id': sample_id,
                'token_ids': token_ids_tensor,  
                'attention_maps': attention_tensor.detach().cpu(),  
                'relative_attention_maps': relative_attention_tensor.detach().cpu() 
            }
            
            return attention_data
            
        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            import traceback
            traceback.print_exc()
            return None


def collect_attention(args, dataloader, model, tokenizer):
    """Collect attention."""
    
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    device = model.device
    dtype = model.dtype
    print(f"Using model dtype: {dtype}")
    
    all_attention_data = []
    
    print("Processing batches with progress bar...")
    for i, batch in enumerate(tqdm(dataloader, desc="Processing batches", unit="batch")):
        # Move batch to the correct device
        input_ids = batch['input_ids'].to(device)
        image_tensor = batch['image'].to(device) if 'image' in batch else None
        image_tensor_aux = batch['image_aux'].to(device) if 'image_aux' in batch else None
        
        if image_tensor is None:
            print("No image found in batch, skipping...")
            continue
            
        image_tensor = image_tensor.to(dtype=dtype)
        image_tensor_aux = image_tensor_aux.to(dtype=dtype)
        # Get unique_id for the batch (batch size is always 1)
        unique_id = batch['unique_id'].to(device)
        sample_id = unique_id.item()
        
        # print(f"\nProcessing sample {sample_id}")
        
        # Find vision token position
        vision_token_start = None
        
        # Look for the image token in the input
        for idx, token_id in enumerate(input_ids[0]):
            if token_id == IMAGE_TOKEN_INDEX:
                vision_token_start = idx
                break
        
        # token_list = []
        # for idx, token_id in enumerate(input_ids[0]):
        #     if token_id != IMAGE_TOKEN_INDEX:
        #         token_list.append(tokenizer.decode([token_id]))
        
        # print(token_list)

        if vision_token_start is None:
            print(f"Warning: No image token found in input for sample {sample_id}, skipping...")
            continue
        
        vision_token_end = vision_token_start + model.get_model().vision_tower.num_patches
        # print(f"Vision token position: [{vision_token_start}, {vision_token_end}]")
        
        # Process sample and collect attention
        attention_data = process_sample_attention(
            model, 
            tokenizer, 
            input_ids, 
            image_tensor, 
            image_tensor_aux,
            sample_id, 
            vision_token_start, 
            vision_token_end, 
            args.output_path
        )
        
        # if successfully processed, add data to the list
        if attention_data is not None:
            all_attention_data.append(attention_data)
    
    # after all samples are processed, save the entire dataset
    if all_attention_data:
        # extract the filename from data_path (without extension)
        import os
        data_filename = os.path.splitext(os.path.basename(args.data_path))[0]
        combined_data_path = f'{args.output_path}/all_attention_data_{data_filename}.pt'
        torch.save(all_attention_data, combined_data_path)
        print(f"All attention data saved to: {combined_data_path}")
        print(f"Processed {len(all_attention_data)} samples")
    else:
        print("No samples were successfully processed, no data to save")
    
    print("Attention collection completed")


def main():
    parser = argparse.ArgumentParser(description="Script for collecting model information")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, default="/data2/jkx/MGM/work_dirs/MGM/MGM-7B", help="Path to the model")
    parser.add_argument("--output_path", type=str, default="/data2/jkx/MGM/output/attention_info", help="Path to save outputs")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, default="/data2/jkx/MGM/output/chunk_json/chunk_1.json", help="Path to the json file")
    parser.add_argument("--image_folder", type=str, default="/data2/jkx/LLaVA/playground/data", help="Path to image folder")
    parser.add_argument("--is_multimodal", action="store_true", help="Whether the data is multimodal")
    parser.add_argument("--image_aspect_ratio", type=str, default="square", help="Image aspect ratio")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    
    # Collection parameters
    parser.add_argument("--info_type", type=str, default="grads", choices=["grads", "reps", "delta", "check_data"], 
                        help="Type of information to collect (check_data for data inspection only)")
    parser.add_argument("--gradient_type", type=str, default="sgd", choices=["sgd", "adam", "sign"], 
                        help="Type of gradient")
    parser.add_argument("--gradient_projection_dimension", type=int, default=5120, 
                        help="Dimension of gradient projection")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples")
    parser.add_argument("--save_interval", type=int, default=160, help="Interval for saving outputs")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--load_4bit", action="store_true", help="Load 4-bit model")
    parser.add_argument("--load_8bit", action="store_true", help="Load 8-bit model")
    
    args = parser.parse_args()
    
    
    # Create data_args object for compatibility with existing code
    class DataArgs:
        def __init__(self):
            self.data_path = args.data_path
            self.lazy_preprocess = False
            self.is_multimodal = args.is_multimodal
            self.image_folder = args.image_folder
            self.image_aspect_ratio = args.image_aspect_ratio
            self.image_processor = None
    
    data_args = DataArgs()
    
    print("Command-line arguments:", args)
    
    # Initialize model
    tokenizer, model, image_processor = initialize_model(args)
    data_args.image_processor = image_processor
    data_args.model_config = model.config
    # Load dataset
    dataset = get_dataset(data_args, args.data_path, tokenizer, args.max_length)
    dataloader = get_dataloader(dataset, tokenizer=tokenizer)
    
    print(f"Dataset loaded from file:", args.data_path)
    print("Number of samples in the dataset:", len(dataset))
    
    # Collect information
    # check_data(args, dataloader, model, tokenizer)
    collect_attention(args, dataloader, model, tokenizer)
    print("Operation completed successfully")


if __name__ == "__main__":
    main()