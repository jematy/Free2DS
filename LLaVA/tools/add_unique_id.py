import json
from tqdm import tqdm
import argparse

def main(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    unique_counter = 1

    for item in tqdm(data, desc="Processing"):
        item["unique_id"] = unique_counter
        unique_counter += 1

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add unique_id to JSON records starting from 1.")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file path")
    args = parser.parse_args()

    main(args.input_file, args.output_file)
