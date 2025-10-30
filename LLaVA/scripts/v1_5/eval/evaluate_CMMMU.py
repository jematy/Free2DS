import json
import argparse

def load_jsonl_to_dict(path):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                qid = obj.get('id') or obj.get('question_id')
                answer = obj.get('answer') or obj.get('text')  # 加这句支持 text 字段
                if qid is not None:
                    data[int(qid)] = answer
    return data

def compute_accuracy(pred_path, answer_path):
    predictions = load_jsonl_to_dict(pred_path)
    # print(predictions)
    answers = load_jsonl_to_dict(answer_path)

    total = 0
    correct = 0
    unmatched = 0

    for qid, correct_ans in answers.items():
        pred_ans = predictions.get(qid)
        total += 1
        # print(correct_ans)
        # print(pred_ans)
        # exit(0)
        if pred_ans is None:
            unmatched += 1
        elif str(pred_ans).strip() == str(correct_ans).strip():
            correct += 1

    print(f"✅ 总题数: {total}")
    print(f"✅ 匹配到的题目数: {total - unmatched}")
    print(f"✅ 正确数: {correct}")
    print(f"✅ 准确率: {correct / total * 100:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="计算模型预测的准确率")
    parser.add_argument("--predictions", help="模型输出文件路径（如 predictions.jsonl）")
    parser.add_argument("--answers", help="标准答案文件路径（如 answers.jsonl）")

    args = parser.parse_args()
    compute_accuracy(args.predictions, args.answers)

if __name__ == "__main__":
    main()
