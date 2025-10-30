import json
import argparse

def load_jsonl(path):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line.strip())
            qid = int(obj['question_id'])
            ans = obj.get('text') or obj.get('answer')
            data[qid] = ans.strip()
    return data

def evaluate_accuracy(predictions, answers):
    total = 0
    correct = 0
    for qid, gold in answers.items():
        qid=qid+1000
        if qid in predictions:
            total += 1
            if predictions[qid].strip().lower() == gold.strip().lower():
                correct += 1
    accuracy = correct / total if total else 0.0
    return correct, total, accuracy

def main():
    parser = argparse.ArgumentParser(description="Evaluate model accuracy.")
    parser.add_argument("--pred_file", help="Path to the model output JSONL file")
    parser.add_argument("--answer_file", help="Path to the ground truth JSONL file")
    args = parser.parse_args()

    preds = load_jsonl(args.pred_file)
    gts = load_jsonl(args.answer_file)
    # print(preds)
    # print(gts)
    correct, total, accuracy = evaluate_accuracy(preds, gts)

    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
