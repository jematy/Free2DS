import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from mgm.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from mgm.conversation import conv_templates, SeparatorStyle
from mgm.model.builder import load_pretrained_model
from mgm.utils import disable_torch_init
from mgm.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image
import requests
from io import BytesIO
import re


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def sanitize_filename(text):
    """
    Sanitize a string to be used as a filename by replacing invalid characters.
    """
    # Replace common invalid filename characters
    invalid_chars = r'[<>:"/\\|?*\x00-\x1F]'
    return re.sub(invalid_chars, '_', text)


def visualize_attention(args):
    

    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    print(args.model_path)
    print(model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name,
        load_4bit=True
    )


    # Determine conversation mode
    if '8x7b' in model_name.lower():
        conv_mode = "mistral_instruct"
    elif '34b' in model_name.lower():
        conv_mode = "chatml_direct"
    elif '2b' in model_name.lower():
        conv_mode = "gemma"
    else:
        conv_mode = "vicuna_v1"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode
        
    conv = conv_templates[args.conv_mode].copy()#默认为vicuna_v1

    # Load the image
    image_tensor = None
    image_tensor_aux = []
    
    image = load_image(args.image_file)

    if args.image_file is not None:
        images = []
        if ',' in args.image_file:
            images = args.image_file.split(',')
        else:
            images = [args.image_file]
        
        image_convert = []
        for _image in images:
            image_convert.append(load_image(_image))
    
        if hasattr(model.config, 'image_size_aux'):
            if not hasattr(image_processor, 'image_size_raw'):
                image_processor.image_size_raw = image_processor.crop_size.copy()
            image_processor.crop_size['height'] = model.config.image_size_aux
            image_processor.crop_size['width'] = model.config.image_size_aux
            image_processor.size['shortest_edge'] = model.config.image_size_aux
        
        # Similar operation in model_worker.py
        image_tensor = process_images(image_convert, image_processor, model.config) #torch.Size([1, 3, 768, 768])
    
        image_grid = getattr(model.config, 'image_grid', 1)
        if hasattr(model.config, 'image_size_aux'):
            raw_shape = [image_processor.image_size_raw['height'] * image_grid,
                        image_processor.image_size_raw['width'] * image_grid]
            image_tensor_aux = image_tensor 
            image_tensor = torch.nn.functional.interpolate(image_tensor,
                                                        size=raw_shape,
                                                        mode='bilinear',
                                                        align_corners=False)
        else:
            image_tensor_aux = []
    
        if isinstance(image_tensor, list):
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
            image_tensor_aux = [image.to(model.device, dtype=torch.float16) for image in image_tensor_aux]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
            if not isinstance(image_tensor_aux, list):
                image_tensor_aux = image_tensor_aux.to(model.device, dtype=torch.float16)
            else:
                image_tensor_aux = []
    else:
        images = None

    # Process the query



    query = args.query

    if args.image_file is not None:
        # Add image tokens to the query
        if model.config.mm_use_im_start_end:
            query = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + query
        else:
            num_images = len(args.image_file.split(',')) if ',' in args.image_file else 1
            query = (DEFAULT_IMAGE_TOKEN + '\n') * num_images + query

    
    print(conv)
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Prepare input tokens
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).to(model.device)

    # Get the starting token position of the image token in the prompt
    vision_token_start = len(tokenizer(prompt.split(DEFAULT_IMAGE_TOKEN)[0], return_tensors='pt')["input_ids"][0])
    vision_token_end = vision_token_start + model.get_model().vision_tower.num_patches

    print("Vision token position: [{}, {}]".format(vision_token_start, vision_token_end))
    token_list = []
    for idx, token_id in enumerate(input_ids[0]):
            if token_id != IMAGE_TOKEN_INDEX:
                token_list.append(tokenizer.decode([token_id]))
        
    print(token_list)
    # Generate response and get attention scores
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            images_aux=image_tensor_aux if len(image_tensor_aux)>0 else None,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            bos_token_id=tokenizer.bos_token_id,  # Begin of sequence token
            eos_token_id=tokenizer.eos_token_id,  # End of sequence token
            pad_token_id=tokenizer.pad_token_id,  # Pad token
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=True,
        )
    
    # Decode the generated response
    outputs = tokenizer.decode(output_ids["sequences"][0], skip_special_tokens=True).strip()
    print("Generated Response:", outputs)

    # Process attention scores
    attention_data = output_ids["attentions"]
    input_token_len = model.get_model().vision_tower.num_patches + len(input_ids[0]) - 1  # -1 for image token

    output_token_len = len(output_ids["sequences"][0])
    output_token_start = input_token_len
    output_token_end = input_token_len + output_token_len
    # Extract output tokens
    output_tokens = []
    for i in range(output_token_len):
        token_id = output_ids["sequences"][0][i].item()
        token = tokenizer.decode([token_id])
        output_tokens.append(token)
    
    print("Output tokens:", output_tokens)
    
    # Process attention scores for each token
    vision_attention_scores = []
    
    for i, attn_step in enumerate(attention_data):
        if i >= len(output_tokens):
            break
        
        token_attn_to_vision = 0.0
        
        for layer_idx, layer_attn in enumerate(attn_step):
            aggregated_attn = layer_attn.mean(dim=1) #[batch_size, heads, seq_len, seq_len]
            layer_attn_score = aggregated_attn[0, -1, vision_token_start:vision_token_end].sum().item()
            token_attn_to_vision += layer_attn_score
        
        token_attn_to_vision /= len(attn_step)
        
        vision_attention_scores.append({
            'token': output_tokens[i+1],
            'attention_score': token_attn_to_vision,
        })
    
    print("\nOutput tokens' attention to vision tokens:")
    for item in vision_attention_scores:
        print(f"Token: '{item['token']}', Attention score: {item['attention_score']:.4f}")
    
    # Create directory for output visualizations
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot overall attention scores
    if len(vision_attention_scores) > 0:
        tokens = [item['token'] for item in vision_attention_scores]
        scores = [item['attention_score'] for item in vision_attention_scores]
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(scores)), scores, marker='o', linestyle='-')
        plt.xticks(range(len(scores)), tokens, rotation=45, ha='right')
        plt.xlabel('Output Tokens')
        plt.ylabel('Attention to Vision Tokens')
        plt.title('Token Attention to Vision')
        plt.tight_layout()
        plt.savefig(f'{args.output_dir}/token_vision_attention.png')
        print(f"Attention score plot saved to {args.output_dir}/token_vision_attention.png")
        
        # Generate heatmaps
        patch_size = model.get_model().vision_tower.config.patch_size
        image_size = model.get_model().vision_tower.config.image_size
        num_patches_per_side = image_size // patch_size
        
        all_attention_maps = []
        valid_tokens = []
        
        # Extract EOS token attention as baseline
        if len(attention_data) > 0:
            # Get the last token's attention (EOS token)
            eos_attn_step = attention_data[-1]
            
            eos_attention_map = torch.zeros((vision_token_end - vision_token_start,), device=model.device)
            
            for layer_idx, layer_attn in enumerate(eos_attn_step):
                layer_attn_aggregated = layer_attn.mean(dim=1)
                layer_attn_score = layer_attn_aggregated[0, -1, vision_token_start:vision_token_end]
                eos_attention_map += layer_attn_score
            
            eos_attention_map /= len(eos_attn_step)
            
            # Reshape EOS attention map (but don't resize yet)
            eos_map = eos_attention_map.reshape(num_patches_per_side, num_patches_per_side)
            
            # Create relative attention maps (before interpolation)
            relative_attention_maps = []
            epsilon = 0  # Small constant to avoid division by zero
            
            # Also generate regular attention maps for comparison
            all_attention_maps = []
            valid_tokens = []
            
            for token_idx in range(len(vision_attention_scores)):
                if token_idx >= len(attention_data):
                    continue
                    
                attn_step = attention_data[token_idx]
                
                token_attention_map = torch.zeros((vision_token_end - vision_token_start,), device=model.device)
                
                for layer_idx, layer_attn in enumerate(attn_step):
                    layer_attn_aggregated = layer_attn.mean(dim=1)
                    
                    layer_attn_score = layer_attn_aggregated[0, -1, vision_token_start:vision_token_end]
                    token_attention_map += layer_attn_score
                
                token_attention_map /= len(attn_step)
                
                # Reshape to 2D grid
                attention_map = token_attention_map.reshape(num_patches_per_side, num_patches_per_side)
                
                # Calculate relative attention (before interpolation)
                rel_attn = attention_map / (eos_map + epsilon)
                
                # Normalize to sum to 1
                if rel_attn.sum() > 0:
                    rel_attn = rel_attn / rel_attn.sum()
                
                # Resize regular attention map
                attention_map_resized = torch.nn.functional.interpolate(
                    attention_map.unsqueeze(0).unsqueeze(0),
                    size=(image.size[1], image.size[0]),
                    mode='bicubic',
                    align_corners=False
                ).squeeze()
                
                # Now resize the normalized relative attention map
                rel_attn_resized = torch.nn.functional.interpolate(
                    rel_attn.unsqueeze(0).unsqueeze(0),
                    size=(image.size[1], image.size[0]),
                    mode='bicubic',
                    align_corners=False
                ).squeeze()
                
                all_attention_maps.append(attention_map_resized.cpu().numpy())
                relative_attention_maps.append(rel_attn_resized.cpu().numpy())
                valid_tokens.append(tokens[token_idx])
            
            # Calculate number of tokens for visualization
            n_tokens = len(relative_attention_maps)
            
            # Create visualization for relative attention maps
            if len(relative_attention_maps) > 0:
                # Grid visualization for relative attention
                grid_size = int(np.ceil(np.sqrt(n_tokens + 1)))
                
                plt.figure(figsize=(grid_size * 4, grid_size * 4))
                
                # Show original image first
                plt.subplot(grid_size, grid_size, 1)
                plt.imshow(image)
                plt.title("Original Image")
                plt.axis('off')
                
                # Show each token's relative heatmap
                for i, (token_text, rel_attention_map) in enumerate(zip(valid_tokens, relative_attention_maps)):
                    plt.subplot(grid_size, grid_size, i + 2)
                    plt.imshow(image)
                    plt.imshow(rel_attention_map, alpha=0.5, cmap='jet')
                    plt.title(f"Relative '{token_text}'")
                    plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(f"{args.output_dir}/combined_relative_attention_heatmap.png", dpi=300)
                print(f"Combined relative attention heatmap saved to {args.output_dir}/combined_relative_attention_heatmap.png")
                
                # Create an organized version with scores
                rows = int(np.ceil((n_tokens) / 4)) + 1  # 4 tokens per row + 1 row for original image
                cols = min(4, n_tokens)
                
                plt.figure(figsize=(cols * 4, rows * 4))
                
                # Original image in larger size
                plt.subplot(rows, cols, 1)
                plt.imshow(image)
                plt.title("Original Image")
                plt.axis('off')
                
                # Show each token's relative heatmap with attention score
                for i, (token_text, rel_attention_map) in enumerate(zip(valid_tokens, relative_attention_maps)):
                    score = vision_attention_scores[i]['attention_score']
                    plt.subplot(rows, cols, i + cols + 1)  # Start in second row
                    plt.imshow(image)
                    plt.imshow(rel_attention_map, alpha=0.5, cmap='jet')
                    plt.title(f"Rel '{token_text}' (score: {score:.4f})")
                    plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(f"{args.output_dir}/organized_relative_attention_heatmap.png", dpi=300)
                print(f"Organized relative attention heatmap saved to {args.output_dir}/organized_relative_attention_heatmap.png")
        
        # Check if there are valid attention maps to display
        if n_tokens == 0:
            print("No valid attention maps to display")
        else:
            # Create individual heatmaps for each token
            """
            for i, (token_text, attention_map) in enumerate(zip(valid_tokens, all_attention_maps)):
                plt.figure(figsize=(8, 8))
                plt.imshow(image)
                plt.imshow(attention_map, alpha=0.5, cmap='jet')
                plt.title(f"Token: '{token_text}'")
                plt.axis('off')
                plt.tight_layout()
                # Comment out individual token heatmap saving
                # sanitized_token = sanitize_filename(token_text.strip())
                # plt.savefig(f"{args.output_dir}/token_{i}_{sanitized_token}.png", dpi=150)
                plt.close()
            """
            
            # Create a grid with all tokens
            grid_size = int(np.ceil(np.sqrt(n_tokens + 1)))  # +1 for original image
            
            plt.figure(figsize=(grid_size * 4, grid_size * 4))
            
            # Show original image first
            plt.subplot(grid_size, grid_size, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis('off')
            
            # Show each token's heatmap
            for i, (token_text, attention_map) in enumerate(zip(valid_tokens, all_attention_maps)):
                plt.subplot(grid_size, grid_size, i + 2)  # +2 because 1 is original image
                plt.imshow(image)
                plt.imshow(attention_map, alpha=0.5, cmap='jet')
                plt.title(f"'{token_text}'")
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{args.output_dir}/combined_attention_heatmap.png", dpi=300)
            print(f"Combined attention heatmap saved to {args.output_dir}/combined_attention_heatmap.png")
            
            # Create a more organized version with attention scores
            rows = int(np.ceil((n_tokens) / 4)) + 1  # 4 tokens per row + 1 row for original image
            cols = min(4, n_tokens)
            
            plt.figure(figsize=(cols * 4, rows * 4))
            
            # Original image in larger size
            plt.subplot(rows, cols, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis('off')
            
            # Show each token's heatmap with attention score
            for i, (token_text, attention_map) in enumerate(zip(valid_tokens, all_attention_maps)):
                score = vision_attention_scores[i]['attention_score']
                plt.subplot(rows, cols, i + cols + 1)  # Start in second row
                plt.imshow(image)
                plt.imshow(attention_map, alpha=0.5, cmap='jet')
                plt.title(f"'{token_text}' (score: {score:.4f})")
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{args.output_dir}/organized_attention_heatmap.png", dpi=300)
            print(f"Organized attention heatmap saved to {args.output_dir}/organized_attention_heatmap.png")
            
            # Remove superimposed attention heatmap code
    
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/data2/jkx/MGM/work_dirs/MGM/MGM-7B")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default="/data2/jkx/LLaVA/2099_1.png")
    parser.add_argument("--query", type=str, default="What's the animal? In ten words")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="attention_heatmaps_rel")
    args = parser.parse_args()

    visualize_attention(args) 