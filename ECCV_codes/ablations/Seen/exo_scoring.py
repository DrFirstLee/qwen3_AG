import os
import sys
import torch
import pandas as pd
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# --- 경로 설정 ---
# 현재 파일 위치 기준
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("/home/bongo/porter_notebook/research/qwen3")

# 모듈 임포트
from config import AGD20K_PATH, model_name
from file_managing import make_input_image

# ------------------------------------------------------
# 1. 환경 및 모델 설정
# ------------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ENABLE_SDPA"] = "1"

INPUT_PKL_PATH = "/home/bongo/porter_notebook/research/qwen3/ECCV_codes/ablations/Seen/2B_all_trials_metrics.pkl"
OUTPUT_PKL_PATH = "/home/bongo/porter_notebook/research/qwen3/ECCV_codes/ablations/Seen/2B_all_trials_scored.pkl"

print(f"📂 Reading DataFrame from: {INPUT_PKL_PATH}")
df = pd.read_pickle(INPUT_PKL_PATH)
print(f"   Total rows: {len(df)}")

print(f"🤖 Loading Model: {model_name}...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map="cuda", 
)
processor = AutoProcessor.from_pretrained(model_name)
device = model.device

# ------------------------------------------------------
# 2. Scoring 함수 정의
# ------------------------------------------------------
def calculate_targeted_score(model, processor, image_path, target_action, object_name):
    # 초기값 설정
    has_text = "N"
    
    # 1. 전처리: 핵심 동사 추출 (예: 'hold_monitor' -> 'hold')
    core_action = target_action.split('_')[0].lower()
    
    # 2. 질문 구성
    query = f"What actions is the person doing with the {object_name}? list all the possible verbs. Only list the verbs."
    
    # 3. 이미지 및 입력 생성
    try:
        image_base64 = make_input_image(str(image_path))
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return 0.0, "image_error", "N"

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": f"data:image/jpeg;base64,{image_base64}"},
            {"type": "text", "text": query}
        ]}
    ]
    
    inputs = processor.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_dict=True, 
        return_tensors="pt"
    ).to(device)
    
    # Vision Token 인덱스 찾기 (입력 프롬프트 내에서)
    input_ids_list = inputs.input_ids[0].tolist()
    vis_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vis_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    
    try:
        vis_start_idx = input_ids_list.index(vis_start_id)
        vis_end_idx = input_ids_list.index(vis_end_id)
    except ValueError:
        # 혹시 vision token이 없는 경우 방어
        return 0.0, "token_error", "N"

    # Inference
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=30, 
            output_attentions=True, 
            return_dict_in_generate=True,
            do_sample=False  # Deterministic 결과
        )
        
    output_ids = generated_ids.sequences[0][inputs.input_ids.shape[1]:]
    full_text = processor.decode(output_ids, skip_special_tokens=True).lower()
    
    # 4. [1차 필터] 문장 전체에 핵심 동사가 없으면 0점
    if core_action not in full_text:
        return 0.0, full_text, has_text

    # 5. Targeted Scoring (Attention Energy Calculation)
    total_vis_score = 0.0     # 매칭된 토큰들의 합
    matched_count = 0         # 매칭된 토큰 수
    
    all_tokens_energy_sum = 0.0 # (Safety Net용)
    valid_token_count = 0       # (Safety Net용)
    
    # generated_ids.attentions는 튜플(생성된 토큰 수)로 구성됨
    # 각 요소는 튜플(레이어 수) -> 텐서(배치, 헤드, 시퀀스, 시퀀스)
    
    for i, token_id in enumerate(output_ids):
        # 인덱스 범위 체크 (생성된 토큰 길이보다 attentions 길이가 짧을 수 있음 - 캐싱 때문)
        if i >= len(generated_ids.attentions): 
            break
            
        token_str = processor.decode([token_id], skip_special_tokens=True).lower().strip()
        if not token_str: continue 

        # --- [에너지 계산] ---
        token_energy = 0.0
        # 모든 레이어의 어텐션 합산
        for layer_attn in generated_ids.attentions[i]:
            # layer_attn shape: [1, num_heads, 1, current_total_seq_len]
            # vision token 영역: vis_start_idx+1 ~ vis_end_idx
            vision_attn = layer_attn[0, :, 0, vis_start_idx+1 : vis_end_idx]
            token_energy += vision_attn.sum().item()
        
        all_tokens_energy_sum += token_energy
        valid_token_count += 1

        # --- [토큰 매칭 확인] ---
        # 1. Core Action이 Token을 포함 OR 2. Token이 Core Action을 포함
        if (core_action in token_str) or (token_str in core_action and len(token_str) > 1): 
            has_text = "Y"
            total_vis_score += token_energy
            matched_count += 1
            
    # [결과 반환 로직]
    if matched_count > 0:
        # 정확히 매칭된 토큰들의 평균 에너지
        final_score = total_vis_score / matched_count
    else:
        # [Safety Net] 텍스트엔 있지만 토큰 매칭 실패 시 -> 전체 문장 평균 에너지
        if valid_token_count > 0:
            final_score = all_tokens_energy_sum / valid_token_count
        else:
            final_score = 0.0

    return final_score, full_text, has_text

# ------------------------------------------------------
# 3. Main Loop
# ------------------------------------------------------
# 결과를 담을 리스트 (딕셔너리 형태가 안전함)
results = []

print("🚀 Starting Scoring Loop...")

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scoring Exo Images"):
    
    # 계산 수행
    score, text, has_text_flag = calculate_targeted_score(
        model, 
        processor, 
        row['exo_path'], 
        row['action'], 
        row['object']
    )
    
    # 결과 저장 (기존 row 데이터를 복사 후 추가)
    new_row = row.to_dict()
    new_row['attn_score'] = score
    new_row['gen_text'] = text
    new_row['has_action_text'] = has_text_flag
    
    results.append(new_row)
    
    # 100개마다 중간 저장 (안전장치)
    if (idx + 1) % 100 == 0:
        temp_df = pd.DataFrame(results)
        temp_df.to_pickle(OUTPUT_PKL_PATH)

# ------------------------------------------------------
# 4. Final Save
# ------------------------------------------------------
final_df = pd.DataFrame(results)
final_df.to_pickle(OUTPUT_PKL_PATH)

print("\n🎉 Scoring Complete!")
print(f"💾 Saved to: {OUTPUT_PKL_PATH}")
print(final_df[['action', 'gen_text', 'attn_score', 'has_action_text']].head())