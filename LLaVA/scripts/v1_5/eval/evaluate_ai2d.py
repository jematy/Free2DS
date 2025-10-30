import json
import argparse
from tqdm import tqdm

def load_jsonl(path):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            qid = str(item['question_id'])
            data[qid] = item
    return data

def evaluate(pred_file, gt_file):
    preds = load_jsonl(pred_file)
    gts = load_jsonl(gt_file)

    total = 0
    correct = 0
    missing = 0

    for qid, gt in gts.items():
        total += 1
        pred = preds.get(qid)
        if pred is None:
            missing += 1
            continue
        pred_answer = str(pred.get('text')).strip().upper()
        gt_answer = str(gt.get('answer')).strip().upper()
        if pred_answer == gt_answer:
            correct += 1

    accuracy = correct / total if total > 0 else 0
    print(f"✅ Total: {total}")
    print(f"✅ Correct: {correct}")
    print(f"❌ Missing predictions: {missing}")
    print(f"🎯 Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate model predictions against ground truth answers.")
    parser.add_argument('--pred_file', type=str, required=True, help='Path to the model output jsonl file')
    parser.add_argument('--gt_file', type=str, required=True, help='Path to the ground truth jsonl file')
    args = parser.parse_args()

    evaluate(args.pred_file, args.gt_file)
