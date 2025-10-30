import json
import os

# Path to the data file
data_file = '/data2/jkx/LLaVA/output/llava_curriculum.json'

# Check if file exists
if not os.path.exists(data_file):
    print(f"File not found: {data_file}")
    print("Please provide the correct path to the conversation data file.")
    exit(1)

# Load the data
try:
    with open(data_file, 'r') as f:
        data = json.load(f)
except json.JSONDecodeError:
    print(f"Error decoding JSON from {data_file}")
    exit(1)
except Exception as e:
    print(f"Error reading file {data_file}: {e}")
    exit(1)

# Count total conversations
total_samples = len(data)
print(f"Total conversation samples: {total_samples}")

# Count conversations with <image> tag in value
image_samples = 0
for item in data:
    has_image = False
    for conv in item['conversations']:
        if '<image>' in conv['value']:
            has_image = True
            break
    if has_image:
        image_samples += 1

# Calculate percentage
percentage = (image_samples / total_samples) * 100 if total_samples > 0 else 0

print(f"Data file: {data_file}")
print(f"Conversations containing '<image>' tag: {image_samples}")
print(f"Percentage of conversations with '<image>' tag: {percentage:.2f}%") 