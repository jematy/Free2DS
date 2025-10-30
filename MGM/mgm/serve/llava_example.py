import os
os.environ['CUDA_VISIBLE_DEVICES'] = "5"
import sys
sys.path.append("./models")
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import cv2
from PIL import Image

import torch
import torch.nn.functional as F

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from utils import (
    load_image, 
    aggregate_llm_attention, aggregate_vit_attention,
    heterogenous_stack,
    show_mask_on_image
)
# ===> specify the model path
model_path = "/data2/jkx/LLaVA/checkpoints/llava-v1.5-7b"

# load the model
load_8bit = False
load_4bit = False
device = "cuda" if torch.cuda.is_available() else "cpu"

disable_torch_init()

model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, 
    None, # model_base
    model_name, 
    load_8bit, 
    load_4bit, 
    device=device
)

# ===> specify the image path or url and the prompt text
# image_path_or_url = "/data2/jkx/LLaVA/IMG_1888.JPG"
# prompt_text = "What's this in ten words"
image_path_or_url = "/data2/jkx/LLaVA/IMG_1888.JPG"
prompt_text = "What's this? in ten words"
# image_path_or_url = "https://kkgithub.com/open-compass/MMBench/blob/main/samples/MMBench/1.jpg?raw=true"
# prompt_text = "What python code can be used to generate the output in the image?"
################################################
# preparation for the generation
# unlikely that you need to change anything here
if "llama-2" in model_name.lower():
    conv_mode = "llava_llama_2"
elif "mistral" in model_name.lower():
    conv_mode = "mistral_instruct"
elif "v1.6-34b" in model_name.lower():
    conv_mode = "chatml_direct"
elif "v1" in model_name.lower():
    conv_mode = "llava_v1"
elif "mpt" in model_name.lower():
    conv_mode = "mpt"
else:
    conv_mode = "llava_v0"

conv = conv_templates[conv_mode].copy()
if "mpt" in model_name.lower():
    roles = ('user', 'assistant')
else:
    roles = conv.roles

image = load_image(image_path_or_url)
image_tensor, images = process_images([image], image_processor, model.config)
image = images[0]
image_size = image.size
if type(image_tensor) is list:
    image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
else:
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

if model.config.mm_use_im_start_end:
    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt_text
else:
    inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt_text

conv.append_message(conv.roles[0], inp)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
# manually removing the system prompt here
# otherwise most attention will be somehow put on the system prompt
prompt = prompt.replace(
    "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. ",
    ""
)

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
################################################

display(image)
print(prompt_text)
print(prompt)

# generate the response
with torch.inference_mode():
    outputs = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[image_size],
        do_sample=False,
        max_new_tokens=512,
        use_cache=True,
        return_dict_in_generate=True,
        output_attentions=True,
    )

text = tokenizer.decode(outputs["sequences"][0]).strip()
print(text)


# constructing the llm attention matrix
aggregated_prompt_attention = []
# x = tokenizer_image_token("<image>", tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
# print(x)
# # print("input_ids", input_ids)
# print(tokenizer("<image>", add_special_tokens=False))
# print(tokenizer.bos_token_id)
# print(tokenizer("Hello", add_special_tokens=False))
print(len(outputs["attentions"]))
for i, layer in enumerate(outputs["attentions"][0]):#先取第一个 batch 样本的所有层注意力，形状变成 (L, H, S, S), 遍历每一层 i=0…L-1，layer 形状是 (H, S, S)。
    # print(layer_attns.shape)
    layer_attns = layer.squeeze(0) #确保 layer_attns 仍是 (H, S, S)。
    
    attns_per_head = layer_attns.mean(dim=0) #对注意力头取平均，得到 (S, S)，表示这一层、所有头综合后的注意力。
    cur = attns_per_head[:-1].cpu().clone()#去掉最后一行,相当于去掉最后一个token，一般是<eos>
    # following the practice in `aggregate_llm_attention`
    # we are zeroing out the attention to the first <bos> token
    # for the first row `cur[0]` (corresponding to the next token after <bos>), however,
    # we don't do this because <bos> is the only token that it can attend to
    cur[1:, 0] = 0. # 屏蔽对 <bos>（index 0）的注意力（除首行外）
    cur[1:] = cur[1:] / cur[1:].sum(-1, keepdim=True) #因为刚才把一些 attention 清零，剩下的权重总和不再是 1，需要按行（每个 token 的注意力分布）重新做归一化。
    aggregated_prompt_attention.append(cur)
aggregated_prompt_attention = torch.stack(aggregated_prompt_attention).mean(dim=0)#平均所有层的 prompt-attention

# llm_attn_matrix will be of torch.Size([N, N])
# where N is the total number of input (both image and text ones) + output tokens
llm_attn_matrix = heterogenous_stack(
    [torch.tensor([1])]
    + list(aggregated_prompt_attention) #Prompt 里第 i 个 token 对整个 Prompt 的注意力分布
    + list(map(aggregate_llm_attention, outputs["attentions"])) #map 是一个内建函数，用来把同一个函数批量地应用到一个可迭代对象（iterable）的每个元素上。
)

# visualize the llm attention matrix
# ===> adjust the gamma factor to enhance the visualization
#      higer gamma brings out more low attention values
gamma_factor = 1
enhanced_attn_m = np.power(llm_attn_matrix.numpy(), 1 / gamma_factor)

fig, ax = plt.subplots(figsize=(10, 20), dpi=150)
ax.imshow(enhanced_attn_m, vmin=enhanced_attn_m.min(), vmax=enhanced_attn_m.max(), interpolation="nearest")

# identify length or index of tokens
input_token_len = model.get_vision_tower().num_patches + len(input_ids[0]) - 1 # -1 for the <image> token
print("prompt:", prompt)
print("len(input_ids[0]):", len(input_ids[0]))
print("prompt[0]:", prompt.split("<image>")[0])
print("prompt[1]:", prompt.split("<image>")[1])
vision_token_start = len(tokenizer(prompt.split("<image>")[0], return_tensors='pt')["input_ids"][0])
print("vision_token_start", vision_token_start, "input_token_len", input_token_len)
print("model.get_vision_tower().num_patches", model.get_vision_tower().num_patches)
vision_token_end = vision_token_start + model.get_vision_tower().num_patches
output_token_len = len(outputs["sequences"][0])#算上了<s>和<\s>
print("output_token_len", output_token_len)
output_token_start = input_token_len
output_token_end = input_token_len + output_token_len

# look at the attention weights over the vision tokens
overall_attn_weights_over_vis_tokens = []
for i, (row, token) in enumerate(
    zip(
        llm_attn_matrix[input_token_len:], 
        outputs["sequences"][0].tolist()
    )
):
    # print(
    #     i + input_token_len, 
    #     f"{tokenizer.decode(token, add_special_tokens=False).strip():<15}", 
    #     f"{row[vision_token_start:vision_token_end].sum().item():.4f}"
    # )

    overall_attn_weights_over_vis_tokens.append(
        row[vision_token_start:vision_token_end].sum().item()
    )

# plot the trend of attention weights over the vision tokens
fig, ax = plt.subplots(figsize=(20, 5))
ax.plot(overall_attn_weights_over_vis_tokens)
ax.set_xticks(range(len(overall_attn_weights_over_vis_tokens)))
# ax.set_xticklabels(
#     [tokenizer.decode(token, add_special_tokens=False).strip() for token in outputs["sequences"][0].tolist()],
#     rotation=75
# )
xticklabels = [
    tokenizer.decode(token, add_special_tokens=False).strip()
    for token in outputs["sequences"][0].tolist()
    if tokenizer.decode(token, add_special_tokens=False).strip() != "<s>"
]
ax.set_xticklabels(xticklabels, rotation=75)#根据issue可能是<s>的问题
ax.set_title("at each token, the sum of attention weights over all the vision tokens");



# connect with the vision encoder attention
# to visualize the attention over the image

# vis_attn_matrix will be of torch.Size([N, N])
# where N is the number of vision tokens/patches
# `all_prev_layers=True` will average attention from all layers until the selected layer
# otherwise only the selected layer's attention will be used
vis_attn_matrix = aggregate_vit_attention(
    model.get_vision_tower().image_attentions,
    select_layer=model.get_vision_tower().select_layer,
    all_prev_layers=True
)
grid_size = model.get_vision_tower().num_patches_per_side

num_image_per_row = 8
image_ratio = image_size[0] / image_size[1]
num_rows = output_token_len // num_image_per_row + (1 if output_token_len % num_image_per_row != 0 else 0)
fig, axes = plt.subplots(
    num_rows, num_image_per_row, 
    figsize=(10, (10 / num_image_per_row) * image_ratio * num_rows), 
    dpi=150
)
plt.subplots_adjust(wspace=0.05, hspace=0.2)

# whether visualize the attention heatmap or 
# the image with the attention heatmap overlayed
vis_overlayed_with_attn = True

# Step 1: 原始 token 列表
tokens = outputs["sequences"][0].tolist()

# Step 2: 筛选掉不参与 attention 的 token（比如 <s>）
output_token_inds = []
decoded_token_texts = []

for i, token in enumerate(tokens):
    decoded = tokenizer.decode(token, add_special_tokens=False).strip()
    if decoded in ["<s>", "</s>", "<pad>"]:
        continue
    output_token_inds.append(i)
    decoded_token_texts.append(decoded)

# Step 3: 开始绘图时确保 i 是基于 filtered 的 token
for i, ax in enumerate(axes.flatten()):
    if i >= len(output_token_inds):
        ax.axis("off")
        continue

    target_token_ind = output_token_inds[i]
    
    # ✅ 防止越界访问 attention 矩阵
    if target_token_ind >= llm_attn_matrix.shape[0]:
        print(f"Skipping token index {target_token_ind}, out of attn bounds")
        ax.axis("off")
        continue

    attn_weights_over_vis_tokens = llm_attn_matrix[target_token_ind][vision_token_start:vision_token_end]
    attn_weights_over_vis_tokens = attn_weights_over_vis_tokens / attn_weights_over_vis_tokens.sum()

    attn_over_image = []
    for weight, vis_attn in zip(attn_weights_over_vis_tokens, vis_attn_matrix):
        vis_attn = vis_attn.reshape(grid_size, grid_size)
        attn_over_image.append(vis_attn * weight)
    attn_over_image = torch.stack(attn_over_image).sum(dim=0)
    attn_over_image = attn_over_image / attn_over_image.max()
    if torch.isnan(attn_over_image).any() or torch.isinf(attn_over_image).any():
        print(f"⚠️ NaN or Inf detected in attn map for token: {decoded_token_texts[i]}")

    attn_over_image = F.interpolate(
        attn_over_image.unsqueeze(0).unsqueeze(0), 
        size=image.size, 
        mode='nearest'
    ).squeeze()

    np_img = np.array(image)[:, :, ::-1]
    img_with_attn, heatmap = show_mask_on_image(np_img, attn_over_image.numpy())
    # img_with_attn, heatmap = show_mask_on_image(np_img, attn_over_image.numpy(), use_jet=True, alpha=0.5)

    ax.imshow(heatmap if not vis_overlayed_with_attn else img_with_attn)
    
    # ✅ 使用 decode 后过滤好的标签
    ax.set_title(
        decoded_token_texts[i],
        fontsize=7,
        pad=1
    )
    ax.axis("off")