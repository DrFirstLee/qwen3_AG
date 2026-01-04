import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import gc
import numpy as np
import pandas as pd
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from file_managing import make_input_image, make_input_image_exo
from config import AGD20K_PATH, model_name

# ------------------------------------------------------
# 1. 환경 설정
# ------------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ENABLE_SDPA"] = "1"

print(f"🤖 {model_name} 모델 로딩중...")

# Qwen3 Model & Processor 로드
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager", # Attention Map 추출을 위해 eager 모드 필수
    device_map="cuda", 
)
processor = AutoProcessor.from_pretrained(model_name)
device = model.device

# ------------------------------------------------------
# 2. 실행부
# ------------------------------------------------------
system_prompt = (
    "You are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language."
)

df_fin = pd.read_pickle("target_df_w_random_exo.pkl")
# 결과를 담을 새로운 컬럼들
df_fin['output_sentence'] = "" 
df_fin['output_attentions'] = "" # 여기에 복잡한 구조의 데이터가 들어갑니다.

print(f"length of Data : {len(df_fin)}")

for index, row in df_fin.iterrows():
    # if (index < 74) or (index > 80):
    #     continue
    object_name = row['object']
    action = row['action']
    filename = row['filename']
    
    
    # 프롬프트 구성
    description = f"""Refer to the second image (exocentric view) for context. 
    Based on the first image (egocentric view), when people perform {action} with {object_name}, which part of the {object_name} is used for '{action}'?
    Answer in one sentence."""


    file_name_real = f"{AGD20K_PATH}/Seen/testset/egocentric/{action}/{object_name}/{filename}"
    exo_file_name_real = row['random_exo_filename'].replace("/home/DATA/AGD20K", AGD20K_PATH)
    print(f"\n{index} >>> {object_name} | {action} | {filename}")
    
    image_base64 = make_input_image_exo(file_name_real)
    exo_image_base64 = make_input_image_exo(exo_file_name_real)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [
            {"type": "image", "image": f"data:image/jpeg;base64,{image_base64}"},
            {"type": "image", "image": f"data:image/jpeg;base64,{exo_image_base64}"},
            {"type": "text", "text": description}, 
        ]}
    ]

    # -------------------------------------------------------
    # STEP 1: Pre-process Inputs
    # -------------------------------------------------------
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True, # 답변 생성을 위해 True로 변경
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(device)

    # -------------------------------------------------------
    # STEP 2: Generate Output Sentence (답변 생성)
    # -------------------------------------------------------
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024, # 필요한 만큼 조절
            do_sample=False,    # Deterministic
            use_cache=True
        )
    
    # 생성된 답변만 디코딩 (입력 부분 제외)
    input_len = inputs.input_ids.shape[1]
    output_ids = generated_ids[0][input_len:] # 순수 생성된 토큰 ID들
    output_text = processor.decode(output_ids, skip_special_tokens=True)
    
    print(f"📝 Output Sentence: {output_text}")
    df_fin.at[index, 'output_sentence'] = output_text

    # -------------------------------------------------------
    # STEP 3: Forward Pass with Full Sequence (Attention 추출)
    # -------------------------------------------------------
    # 생성된 전체 시퀀스(Input + Output)를 다시 모델에 넣어 Attention Map을 구함
    # QwenVL은 image inputs(pixel_values 등)이 필요하므로 inputs 정보 재사용
    
    full_input_ids = generated_ids # [1, seq_len]
    
    # inputs에 있는 이미지 관련 텐서들을 그대로 사용하되, input_ids만 전체 시퀀스로 교체
    # 주의: Qwen3VLForConditionalGeneration의 forward 인자에 맞춰 전달
    with torch.no_grad():
        outputs = model(
            input_ids=full_input_ids,
            pixel_values=inputs.pixel_values,
            image_grid_thw=inputs.image_grid_thw,
            attention_mask=torch.ones_like(full_input_ids), # 전체 다 보이게
            output_attentions=True,
            return_dict=True
        )
    
    attentions = outputs.attentions # Tuple of (batch, num_heads, seq_len, seq_len)
    
    # -------------------------------------------------------
    # STEP 4: Index Parsing
    # -------------------------------------------------------
    ids_list = full_input_ids[0].tolist()
    tok = processor.tokenizer
    
    # Vision Token ID 정의
    vision_start_id = tok.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id   = tok.convert_tokens_to_ids("<|vision_end|>")

    # 1. 시퀀스 내의 '모든' 이미지 시작/끝 위치 찾기
    all_starts = [i for i, x in enumerate(ids_list) if x == vision_start_id]
    all_ends   = [i for i, x in enumerate(ids_list) if x == vision_end_id]

# [수정] 이미지 2개가 정상적으로 있는지 확인
    if len(all_starts) < 2:
        print("Error: Expected 2 images, but found fewer.")
        raise Exception("Image not found in sequence.")
    else:
        # 첫 번째 이미지 (Egocentric) 범위
        idx1_start, idx1_end = all_starts[0], all_ends[0]
        # 두 번째 이미지 (Exocentric) 범위
        idx2_start, idx2_end = all_starts[1], all_ends[1]

    # 히트맵 생성을 위한 Target(Ego) 설정
    target_start = idx1_start
    target_end = idx1_end

    # Grid Info (첫 번째 이미지 기준)
    grid_t, grid_h, grid_w = inputs.image_grid_thw[0].detach().cpu().numpy()
    llm_grid_h = grid_h // 2
    llm_grid_w = grid_w // 2
    query_start_idx = input_len
    query_end_idx = len(ids_list)  
    output_attn_data = []

    # 각 Output Token에 대해 순회 (Query Iteration)
    for q_idx in range(query_start_idx, query_end_idx):
        token_id = ids_list[q_idx]
        token_str = tok.decode([token_id])
        
        token_data = {
            "token_idx_in_seq": q_idx,
            "token_str": token_str,
            "token_id": token_id,
            "attentions": [] # 각 레이어/헤드의 heatmap 정보
        }

        # 모든 Layer 순회
        for layer_idx, layer_attn in enumerate(attentions):
            # layer_attn shape: [batch, num_heads, seq_len, seq_len]
            # [0, :, q_idx, :] -> 현재 토큰(q_idx)이 전체 시퀀스를 바라보는 Attention
            
            heads_attn = layer_attn[0, :, q_idx, :] # [num_heads, seq_len]
            num_heads = heads_attn.shape[0]

            for head_idx in range(num_heads):
                this_head_attn = heads_attn[head_idx] # [seq_len]
                
                # [수정] 1번 이미지 Attention 합
                attn_img1 = this_head_attn[idx1_start + 1 : idx1_end]
                sum_img1 = float(attn_img1.sum().detach().cpu().item())
                # [수정] 2번 이미지 Attention 합 (존재할 경우)
                sum_img2 = 0.0
                attn_img2 = this_head_attn[idx2_start + 1 : idx2_end]
                sum_img2 = float(attn_img2.sum().detach().cpu().item())
                # [최종] 두 이미지의 Attention 총합
                total_s_img_val = sum_img1 + sum_img2

                # Heatmap 저장 (메모리 절약을 위해 float16 등으로 변환 고려 가능)
                heatmap_np = attn_img1.reshape(llm_grid_h, llm_grid_w).float().cpu().numpy()
                
                # 필요한 정보만 저장 (전체 맵을 다 저장하면 용량이 매우 큽니다!)
                # 여기서는 요청대로 "V 값을 모두 저장" 하도록 heatmap을 저장합니다.
                token_data["attentions"].append({
                    "layer": layer_idx,
                    "head": head_idx,
                    "s_img": total_s_img_val, # <--- 여기가 수정됨 (1+2 합)
                    "s_img_ego": sum_img1,    # (옵션) 나중을 위해 분리해서 저장해도 좋음
                    "s_img_exo": sum_img2,    # (옵션) 
                    "heatmap": heatmap_np 
                })
        
        output_attn_data.append(token_data)

    df_fin.at[index, 'output_attentions'] = output_attn_data
    
    # -------------------------------------------------------
    # STEP 6: Save & Cleanup
    # -------------------------------------------------------
    save_every = 5 # 용량이 크므로 더 자주 저장 권장
    if (index % save_every == 0) and (index > 1):
        save_index = int(index/save_every)
        print(f"✅ Saving!!!=== at index={index}")
        df_fin.iloc[index-5:index].to_pickle(f"exo_attention_result_32B_2_{save_index}.pkl")
        print(f"✅ Saved at index={index} // {save_index}")

    # Memory Cleanup
    del generated_ids, full_input_ids, outputs, attentions, output_attn_data
    torch.cuda.empty_cache()
    gc.collect()

print("작업 완료")