#  nohup python -u exo_test.py >> 2b_selection.log 2>&1 & 

import os
import sys
import torch
import pandas as pd
from tqdm import tqdm
import random
from pathlib import Path
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# --- 경로 설정 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append("/home/bongo/porter_notebook/research/qwen3")

# config 및 file_managing에서 필요한 변수/함수 임포트
from config import AGD20K_PATH, model_name
from file_managing import make_input_image

# ------------------------------------------------------
# 1. 모델 로딩
# ------------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ENABLE_SDPA"] = "1"

print(f"🤖 Loading {model_name} for selection...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map="cuda", 
)
processor = AutoProcessor.from_pretrained(model_name)
device = model.device

# ------------------------------------------------------
# 2. Scoring 함수 (Simple Listing & Targeted Scoring)
# ------------------------------------------------------
def calculate_targeted_score(model, processor, image_path, target_action, object_name):
    has_text = "N"
    # -----------------------------------------------------------
    # 1. 전처리: 핵심 동사 추출
    # -----------------------------------------------------------
    # 사전 없이 단순 split만 사용
    core_action = target_action.split('_')[0].lower() 
    
    # -----------------------------------------------------------
    # 2. 질문: 동사 나열 유도
    # -----------------------------------------------------------
    query = f"What actions is the person doing with the {object_name}? list all the possible verbs. Only list the verbs."
    
    # 3. Inference

    image_base64 = make_input_image(str(image_path))


    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": f"data:image/jpeg;base64,{image_base64}"},
            {"type": "text", "text": query}
        ]}
    ]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
    
    # Vision Token 인덱스 찾기
    input_ids_list = inputs.input_ids[0].tolist()
    vis_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vis_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

    vis_start_idx = input_ids_list.index(vis_start_id)
    vis_end_idx = input_ids_list.index(vis_end_id)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=30, output_attentions=True, return_dict_in_generate=True)
        
    output_ids = generated_ids.sequences[0][inputs.input_ids.shape[1]:]
    full_text = processor.decode(output_ids, skip_special_tokens=True).lower()
    
    # 4. [1차 필터] 문장 전체에 핵심 동사가 없으면 0점 (예: carry 찾는데 walk만 있음)
    if core_action not in full_text:
        return 0.0, full_text, has_text

    # -----------------------------------------------------------
    # 5. Targeted Scoring (Safety Net 추가)
    # -----------------------------------------------------------
    total_vis_score = 0.0     # 매칭된 토큰들의 합
    matched_count = 0         # 매칭된 토큰 수
    
    all_tokens_energy_sum = 0.0 # (Safety Net용) 전체 토큰 에너지 합
    valid_token_count = 0       # (Safety Net용) 전체 유효 토큰 수
    
    for i, token_id in enumerate(output_ids):
        token_str = processor.decode([token_id], skip_special_tokens=True).lower().strip()
        if not token_str: continue # 공백 토큰 무시

        # --- [에너지 계산] ---
        # 매칭 여부와 상관없이 일단 현재 토큰의 에너지를 계산해둡니다.
        token_energy = 0.0
        for layer_attn in generated_ids.attentions[i]:
            # Vision Token 영역만 슬라이싱하여 합산
            vision_attn = layer_attn[0, :, 0, vis_start_idx+1 : vis_end_idx]
            token_energy += vision_attn.sum().item()
        
        # Safety Net을 위해 전체 누적
        all_tokens_energy_sum += token_energy
        valid_token_count += 1

        # --- [토큰 매칭 확인] ---
        # 1. Core Action이 Token을 포함 (예: carry >= car) -> BPE 파편화 대응
        # 2. Token이 Core Action을 포함 (예: carrying >= carry) -> 변형 대응
        if (core_action in token_str) or (token_str in core_action and len(token_str) > 1): 
            has_text = "Y"
            # len > 1 조건: 'c', 'a' 같은 너무 짧은 파편이 엄한 단어에 매칭되는 것 방지
            total_vis_score += token_energy
            matched_count += 1
            
    # -----------------------------------------------------------
    # [결과 반환 로직]
    # -----------------------------------------------------------
    if matched_count > 0:
        # 1. 정확히 매칭된 토큰이 있으면 그 점수 사용 (Best)
        final_score = total_vis_score / matched_count
    else:
        # 2. [Safety Net] 매칭된 토큰은 없지만, full_text에는 정답이 있었음!
        # 토크나이징 문제로 판단하고, 전체 문장의 평균 에너지를 반환 (Fallback)
        if valid_token_count > 0:
            final_score = all_tokens_energy_sum / valid_token_count
        else:
            return 0.0, full_text,has_text # 토큰이 없으면 0점

    return final_score, full_text, has_text


# ------------------------------------------------------
# 3. Main Selection Loop
# ------------------------------------------------------
EXO_ROOT = Path(f"{AGD20K_PATH}/Seen/trainset/exocentric")
valid_ext = {'.jpg', '.jpeg', '.png'}
save_path = "selected_best_exo_images.pkl"

selection_db = []

print(f"📂 Scanning directory: {EXO_ROOT}")

# Action 폴더 순회
actions = sorted([d for d in EXO_ROOT.iterdir() if d.is_dir()])

for action_dir in tqdm(actions, desc="Actions"):

    action_name = action_dir.name
    
    # Object 폴더 순회
    objects = sorted([d for d in action_dir.iterdir() if d.is_dir()])
    
    for obj_dir in objects:
        obj_name = obj_dir.name
        print(f"action: {action_name}, object: {obj_name}")
        # 해당 폴더의 모든 이미지 가져오기
        all_exo_images = [p for p in obj_dir.rglob("*") if p.suffix.lower() in valid_ext]
        
        if len(all_exo_images) == 0:
            print(f"⚠️ No images in {action_name}/{obj_name}")
            continue
            
        best_score = -1.0
        best_image_info = None
        
        # 각 이미지 평가 (Competitive Selection)
        num_samples = min(len(all_exo_images), 20)
        for img_path in random.sample(all_exo_images, num_samples):
            
            score, text,has_text = calculate_targeted_score(model, processor, img_path, action_name, obj_name)
            print(f"img_path: {img_path}, score: {score}, text: {text} / has_text : {has_text}")
            
            # 최고 점수 갱신
            if score > best_score:
                best_score = score
                best_image_info = {
                    "action": action_name,
                    "object": obj_name,
                    "best_exo_path": str(img_path), # 전체 경로 저장
                    "filename": img_path.name,
                    "score": score,
                    "output_text": text
                }
        
        # 결과 저장 (점수가 0이어도 가장 나은 게 없다면 기록되거나, 필터링 가능)
        if best_image_info and best_score > 0:
            selection_db.append(best_image_info)
            # print(f"   ✅ Selected for {action_name}-{obj_name}: {best_image_info['filename']} (Score: {best_score:.2f})")
        else:
            print(f"   ❌ Failed to select for {action_name}-{obj_name} (No valid action detected)")
    # break
# ------------------------------------------------------
# 4. 결과 저장
# ------------------------------------------------------
df_selected = pd.DataFrame(selection_db)
df_selected.to_pickle(save_path)

print("\n" + "="*50)
print(f"🎉 Selection Complete! Saved to {save_path}")
print(f"Total Pairs Processed: {len(df_selected)}")
print("="*50)

# 결과 미리보기
print(df_selected.head())