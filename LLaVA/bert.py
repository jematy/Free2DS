import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import argparse


def load_data(data_path):
    """
    Load the conversation data from a JSON file.
    
    Args:
        data_path: Path to the JSON file containing conversation data
        
    Returns:
        List of dictionaries containing conversation data
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def extract_human_messages(data):
    """
    Extract all human messages from the conversations.
    
    Args:
        data: List of dictionaries containing conversation data
        
    Returns:
        Dictionary mapping unique_ids to lists of human messages and their positions
    """
    human_messages = {}
    
    for item in data:
        unique_id = item.get('unique_id')
        if unique_id is None:
            continue
            
        messages = []
        positions = []
        msg_position = 0
        
        for conversation in item['conversations']:
            if conversation['from'].lower() == 'human':
                # Remove image token if present
                message = conversation['value']
                message = message.replace('<image>', '').replace('<Image>', '')
                message = message.strip()
                messages.append(message)
                positions.append(msg_position)
            msg_position += 1
        
        if messages:  # Only add if there are human messages
            human_messages[unique_id] = {
                'messages': messages,
                'positions': positions
            }
                
    return human_messages


def get_bert_embeddings(messages_data, batch_size=32, num_gpus=1):
    """
    Get BERT embeddings for all messages.
    
    Args:
        messages_data: Dictionary mapping unique_ids to dictionaries with messages and positions
        batch_size: Batch size for processing
        num_gpus: Number of GPUs to use
        
    Returns:
        Dictionary with BERT CLS embeddings for each message
    """
    # Load BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('/data2/jkx/LLaVA/checkpoints/bert-base-uncased')
    model = BertModel.from_pretrained('/data2/jkx/LLaVA/checkpoints/bert-base-uncased')
    
    # Check GPU availability and setup
    if torch.cuda.is_available():
        if num_gpus > 1:
            device = torch.device('cuda')
            # Use DataParallel to distribute across multiple GPUs
            model = torch.nn.DataParallel(model, device_ids=list(range(min(num_gpus, torch.cuda.device_count()))))
            # Increase batch size proportionally to the number of GPUs
            batch_size *= min(num_gpus, torch.cuda.device_count())
            print(f"Using {min(num_gpus, torch.cuda.device_count())} GPUs with batch size {batch_size}")
        else:
            device = torch.device('cuda')
            print("Using single GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    model = model.to(device)
    model.eval()
    
    # Collect all messages across all conversations
    all_texts = []
    all_metadata = []
    
    print("Collecting all messages for batch processing...")
    for uid, data in messages_data.items():
        message_list = data['messages']
        position_list = data['positions']
        
        for i, (message, position) in enumerate(zip(message_list, position_list)):
            all_texts.append(message)
            all_metadata.append({
                'unique_id': uid,
                'conversation_position': position,
                'message': message
            })
    
    print(f"Total collected messages: {len(all_texts)}")
    
    # Process all messages in batches
    all_embeddings = []
    
    for i in tqdm(range(0, len(all_texts), batch_size), desc="Processing batches"):
        batch_texts = all_texts[i:i + batch_size]
        
        # Tokenize
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use [CLS] token embedding as sentence representation
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.extend(batch_embeddings)
    
    return {
        'embeddings': np.array(all_embeddings),
        'metadata': all_metadata
    }


def save_embeddings(embeddings_data, output_path):
    """
    Save the embeddings and metadata to files.
    
    Args:
        embeddings_data: Dictionary with embeddings and metadata
        output_path: Base path for saving files
    """
    # Save embeddings as numpy array
    embeddings_file = output_path + '_embeddings.npy'
    np.save(embeddings_file, embeddings_data['embeddings'])
    print(f"Saved {len(embeddings_data['embeddings'])} embeddings to {embeddings_file}")
    
    # Save metadata as JSON
    metadata_file = output_path + '_metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(embeddings_data['metadata'], f, ensure_ascii=False, indent=2)
    print(f"Saved metadata to {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description='Extract BERT CLS token embeddings from conversation data')
    parser.add_argument('--data_path', type=str, default='/data2/jkx/LLaVA/playground/data/llava_v1_5_mix665k_with_unique_id.json', help='Path to the JSON file with conversation data')
    parser.add_argument('--output_path', type=str, default='bert_embeddings', help='Base path for output files')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for BERT processing')
    parser.add_argument('--num_gpus', type=int, default=7, help='Number of GPUs to use')
    args = parser.parse_args()
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {args.data_path}...")
    data = load_data(args.data_path)
    print(f"Loaded {len(data)} conversation items")
    
    print("Extracting human messages...")
    human_messages = extract_human_messages(data)
    print(f"Extracted messages from {len(human_messages)} conversations")
    
    print("Generating BERT embeddings...")
    embeddings_data = get_bert_embeddings(human_messages, batch_size=args.batch_size, num_gpus=args.num_gpus)
    
    print("Saving embeddings and metadata...")
    save_embeddings(embeddings_data, args.output_path)
    
    print("Extraction complete!")


if __name__ == "__main__":
    main()
