import os
import sys
import re
import io
import json
import gc
import random
import base64
from io import BytesIO

# 데이터 처리 및 시각화
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# 딥러닝 관련
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# ------------------------------------------------------
# 1. 환경 변수 설정 (최상단 배치)
# ------------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ENABLE_SDPA"] = "1"

# ------------------------------------------------------
# 2. 시스템 경로 및 로컬 모듈 로드
# ------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from file_managing import (
    load_selected_samples,
    get_actual_path,
    get_gt_path,
    make_input_image
)
from config import AGD20K_PATH, model_name
from VLM_model_dot_relative import QwenVLModel, MetricsTracker

# ------------------------------------------------------
# 3. 모델 및 프로세서 설정
# ------------------------------------------------------
print(f"🤖 {model_name} 모델 로딩중...")

# Qwen3 Model & Processor 로드
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map="cuda", 
    do_sample=False,       # Greedy Search
    temperature=0.0,
    top_p=1.0,
    num_beams=1,
)
processor = AutoProcessor.from_pretrained(model_name)

device = model.device

# ------------------------------------------------------
# 4. 프롬프트 및 변수 초기화
# ------------------------------------------------------
system_prompt = (
    "You are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language."
)

cnt_d = 0

# --- 실행부 시작 ---

df_fin = pd.read_pickle("target_df_w_description.pkl")
df_fin['s_img'] = ""
# df_fin = df_fin.iloc[3:].reset_index(drop=True)
print(f"length of Data : {len(df_fin)}")

for index, row in df_fin.iterrows():
    object_name = row['object']
    action = row['action']
    filename = row['filename']
    # description = row['description']
    description = f"""When people perform {action} with {object_name}, which part of the {object_name} is used for '{action}'?
                    answer in one sentence."""

    file_name_real = f"{AGD20K_PATH}/Seen/testset/egocentric/{action}/{object_name}/{filename}"
    print(index,' >>>>>>>>>>>>>>>>>>>>>>>   ', object_name,action,filename)
    image_base64 = make_input_image(file_name_real)
    messages = [
                {
                "role": "system", 
                "content": [
                    {"type": "text", "text": system_prompt}
                ]
                },
                {
                "role": "user",
                "content": [
                     {"type": "image", "image": f"data:image/jpeg;base64,{image_base64}"},
                    {"type": "text", "text": description}, 
                   
                            ]
                }
                ]
    # 2. Inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    # Inference: Generation of the output

    with torch.no_grad():
            fw_outputs = model(
            **inputs,
            output_attentions=True,
            use_cache=False,
            return_dict=True
        )
    attentions = fw_outputs.attentions

    # -----------------------
    ids_list = inputs.input_ids[0].tolist()
    tok = processor.tokenizer

    vision_start_id = tok.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id   = tok.convert_tokens_to_ids("<|vision_end|>")

    vis_start_idx = ids_list.index(vision_start_id)
    vis_end_idx   = ids_list.index(vision_end_id)

    q_txt_idx = len(ids_list) - 1

    special_ids = set(tok.all_special_ids)
    def is_whitespace_token(tok_str: str) -> bool:
        if tok_str in ["Ċ", "ĊĊ"]:   # newline / double newline 
            return True
        s = tok_str.replace("▁", "").replace("Ġ", "")
        return s.strip() == ""

    # q_txt is real last token
    q_txt_idx = len(ids_list) - 1
    
    while q_txt_idx > vis_end_idx:
        tid = ids_list[q_txt_idx]
        tstr = tok.convert_ids_to_tokens([tid])[0]

        if tid in special_ids or is_whitespace_token(tstr):
            q_txt_idx -= 1
            continue
        break

    print("vis_start_idx:", vis_start_idx, "vis_end_idx:", vis_end_idx)
    print("q_txt_idx:", q_txt_idx, "/ token:", repr(tok.convert_ids_to_tokens([ids_list[q_txt_idx]])[0]))
    print("decoded:", repr(tok.decode([ids_list[q_txt_idx]])))


    # for j in range(q_txt_idx-10, q_txt_idx):
    #     tid = ids_list[j]
    #     print("check prompts",j, repr(tok.convert_ids_to_tokens([tid])[0]))

    grid_t, grid_h, grid_w = inputs.image_grid_thw[0].detach().cpu().numpy()

    # ✨ 핵심 수정: Qwen3-VL은 2x2 풀링을 하므로 그리드 크기를 2로 나눕니다.
    llm_grid_h = grid_h // 2
    llm_grid_w = grid_w // 2
    num_image_tokens = llm_grid_h * llm_grid_w

    print(f"grid_h : {grid_h}, grid_w : {grid_w}, grid_t :{grid_t}, llm_grid_h : {llm_grid_h}, llm_grid_w : {llm_grid_w}")
    
    input_ids = inputs.input_ids[0].cpu().tolist()

    current_img_size = 1000 
    
    valid_heads_count = 0
    accumulated_heatmap = np.zeros((llm_grid_h, llm_grid_w), dtype=np.float32)
    # --- S_img 저장용 ---
    s_img_list = []  # 각 원소: dict(layer, head, S_img)

    # 4. Layer & Head iteration
    for layer_idx, layer_attn in enumerate(attentions):
        heads_attn = layer_attn[0, :, q_txt_idx, :]   
        # batch : 0, allhead, 
        # query = last text token, 
        # key : all
        num_heads = heads_attn.shape[0]

        for head_idx in range(num_heads):
            this_head_attn = heads_attn[head_idx]     # [seq_len]
            # vision token range(from vis_start_idx to vis_end_idx)
            img_attn_1d = this_head_attn[vis_start_idx + 1 : vis_end_idx]  # ✅ key/value = 이미지 토큰
            # Calculate S_img (sum of image token attention) like CVPR25
            S_img = float(img_attn_1d.sum().detach().cpu().item())
            val = img_attn_1d.sum()

            s_img_list.append({ 
                "layer": layer_idx,
                "head": head_idx,
                "S_img": S_img,
                "heatmap": img_attn_1d.reshape(llm_grid_h, llm_grid_w).float().cpu().numpy()
            })

            total = this_head_attn.sum().item()

            # ## Validation >> total must be 1. OKAY
            # S_img = img_attn_1d.sum().item()
            # S_txt = (this_head_attn[:vis_start_idx].sum() + this_head_attn[vis_end_idx+1:].sum()).item()
            # print(total, S_img, S_txt, S_img + S_txt)

    df_fin.at[index, 's_img'] = s_img_list
    save_every = 20
    if (index ) % save_every == 0:
        df_fin.to_pickle("attention_result_delme.pkl")
        print(f"✅ saved at index={index}")
    del fw_outputs 
    del inputs 
    torch.cuda.empty_cache() 
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    del  attentions, s_img_list


print(df_fin)

    