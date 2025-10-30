#!/usr/bin/env python3
"""
合并filtered_data目录中所有带有chunk的JSON文件
"""
import os
import json
import argparse
from pathlib import Path
import glob


def merge_chunk_files(input_dir, output_file):
    """
    合并指定目录中所有带有chunk的JSON文件
    
    Args:
        input_dir: 输入目录路径，包含要合并的JSON文件
        output_file: 输出JSON文件路径
    """
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"输入目录 {input_dir} 不存在")
    
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取所有包含"chunk"的JSON文件
    chunk_files = glob.glob(os.path.join(input_dir, "*chunk*.json"))
    
    # 排除可能的统计文件或其他非数据文件
    chunk_files = [f for f in chunk_files if "filter_stats" not in os.path.basename(f)]
    
    if not chunk_files:
        raise ValueError(f"在 {input_dir} 中没有找到包含'chunk'的JSON文件")
    
    print(f"找到 {len(chunk_files)} 个chunk文件需要合并")
    
    # 合并所有数据
    merged_data = []
    total_samples = 0
    
    for file_path in sorted(chunk_files):
        file_name = os.path.basename(file_path)
        print(f"正在处理: {file_name}")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # 确保数据是列表格式
            if not isinstance(data, list):
                if isinstance(data, dict):
                    data = list(data.values())
                else:
                    print(f"警告: {file_name} 中的数据不是列表或字典格式，跳过")
                    continue
            
            # 跳过空文件
            if len(data) == 0:
                print(f"警告: {file_name} 为空，跳过")
                continue
                
            file_samples = len(data)
            merged_data.extend(data)
            total_samples += file_samples
            print(f"  - 添加了 {file_samples} 个样本")
            
        except json.JSONDecodeError:
            print(f"警告: {file_name} 不是有效的JSON文件，跳过")
            continue
        except Exception as e:
            print(f"处理 {file_name} 时出错: {str(e)}")
            continue
    
    print(f"\n合并完成! 总共 {total_samples} 个样本")
    
    # 保存合并后的数据
    print(f"正在保存合并后的数据到: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"保存完成! 文件大小: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    return len(merged_data)


def main():
    parser = argparse.ArgumentParser(description="合并filtered_data目录中所有带有chunk的JSON文件")
    parser.add_argument("--input_dir", "-i", type=str, default="/data2/jkx/LLaVA/filtered_data", 
                        help="输入目录路径，包含要合并的JSON文件")
    parser.add_argument("--output", "-o", type=str, default="/data2/jkx/LLaVA/filtered_data/merged_chunks.json", 
                        help="输出JSON文件路径")
    
    args = parser.parse_args()
    
    # 执行合并
    merge_chunk_files(args.input_dir, args.output)


if __name__ == "__main__":
    main() 