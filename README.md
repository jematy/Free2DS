# Free²DS: A Training-Free and Label-Free Data Selection Framework for Visual Instruction Tuning.

Official implementation of **“Free²DS: A Training-Free and Label-Free Data Selection Framework for Visual Instruction Tuning.”**

---


## ⚙️ Environment Setup

We use three environments for modular isolation:

```bash
conda env create -f environment_llava.yml
conda env create -f environment_faiss.yml
conda env create -f environment_mgm.yml
```

## 📦 Dataset Preparation

### Step 1. Download Data and Pretraining Models

1. Follow the [LLaVA](https://github.com/haotian-liu/LLaVA) repository to obtain **LLaVA-665K**.  
2. Download image data as instructed in the LLaVA repo and place it in `LLaVA/playground/`.  
3. Download the **MGM model** according to the original [MGM](https://github.com/dvlab-research/MGM) repository requirements.  
4. Download **BERT** and place it under `./LLaVA/checkpoints/`.  

💡 We also release a 20% subset of the dataset selected by our method.[LLaVA-FDS-133K](https://drive.google.com/file/d/1H7IfsKgqY4t6jD7Pkqj76ypsBOtRvKAE/view?usp=sharing). You can directly use this dataset for training to skip the data selection process.

To improve processing efficiency, we split the dataset into smaller files for parallel processing across multiple GPUs.
### Step 2. Add Unique IDs
Add a unique identifier to each data sample:
```
python tools/add_unique_id.py \
  --input_file ./playground/data/llava_v1_5_mix665k.json \
  --output_file ./playground/data/llava_v1_5_mix665k_with_unique_id.json
```
### Step 3. Split Dataset for Parallel Processing

```
python tools/spilt.py \
  --input_file ./playground/data/llava_v1_5_mix665k_with_unique_id.json \
  --output_dir ./output/chunk_json \
  --num-splits 200
```

## 🎯 Selection
Free²DS consists of two complementary stages — image-side selection and text-side selection.

### 🖼️ Image-side Selection

Step 1. Obtain LLaVA Attention
```
cd LLaVA
conda activate llava
python run_obtain_attention_one_word.py
```

Step 2. Obtain MGM Attention
```
cd MGM
conda activate mgm
python MGM/mgm/serve/run_obtain_attention_lora_one_word.py
```
These steps extract multimodal attention features for each instruction–image pair.

### 💬 Text-side Selection
Step 3. Extract BERT Features
```
cd LLaVA
conda activate faiss
python bert.py
```

Step 4. Compute FAISS-based Clustering and Scoring
```
python bert_clustering_scoring.py
```
This stage measures text-level semantic redundancy and selects diverse, representative samples.

### 🧩 Joint Score Based Selection
Step 5. Combine and Filter Final Dataset

After obtaining both image-side and text-side scores, run the following script to generate the final selected dataset:
```
cd LLaVA
python filter_data_by_attention_iou_and_bert_all.py
```
The resulting filtered dataset will be saved in:
```
./output
```

## 🧠 Training

We follow the official [LLaVA](https://github.com/haotian-liu/LLaVA) training pipeline and fine-tune using only the selected subset.
Before training, make sure to download all required checkpoints as described in the [LLaVA documentation](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md).
```
cd LLaVA
conda activate llava
bash ./scripts/v1_5/finetune_lora.sh
```

## 🚀 Inference & Evaluation

After training on the selected data, you can perform evaluation using following method:

[LLaVA Official Evaluation Script](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md) — standard pipeline for visual instruction tuning benchmarks.

## 🙏 Acknowledgements

Thanks to following open-source works:

- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [MGM](https://github.com/dvlab-research/MGM)
- [BERT (Google)](https://huggingface.co/google-bert/bert-base-uncased)