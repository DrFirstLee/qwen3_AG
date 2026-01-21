#  nohup python -u unseen_all.py >> 3b_all.log 2>&1 & 
import os
import sys
import gc
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import zoom
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, CLIPSegProcessor, CLIPSegForImageSegmentation

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from file_managing import (
    load_selected_samples,
    get_actual_path,
    get_gt_path,
    load_ground_truth,
    prompt_dict_obj,
    get_clipseg_heatmap,
    calculate_metrics,
    prompt_dict_obj,
    make_input_image
)

from VLM_model_dot_relative import MetricsTracker
metrics_tracker_alloutput = MetricsTracker(name="all_output")

from config import AGD20K_PATH, model_name

# ------------------------------------------------------
# 1. 환경 설정 및 모델 로딩
# ------------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ENABLE_SDPA"] = "1"

TARGET_ROOT = f"{AGD20K_PATH}/Unseen/testset/egocentric"
GT_ROOT = f"{AGD20K_PATH}/Unseen/testset/GT" # GT 경로 (환경에 맞게 확인 필요)
SAVE_FILENAME = "attention_result_unseen_final.pkl"
VIS_DIR = "result_vis" # 시각화 저장 폴더
os.makedirs(VIS_DIR, exist_ok=True)

print(f"🤖 {model_name} (Qwen) 로딩중...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map="cuda", 
)
processor = AutoProcessor.from_pretrained(model_name)
device = model.device

print(f"👁️ CLIPSeg (Object Mask용) 로딩중...")
clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

# ------------------------------------------------------
# 2. Helper Functions (Metrics & Utils)
# ------------------------------------------------------
def min_max_normalize(map_data):
    """0~1 정규화"""
    m_min, m_max = map_data.min(), map_data.max()
    if m_max - m_min == 0: return map_data
    return (map_data - m_min) / (m_max - m_min)

def get_clipseg_mask(image_path, text_prompt, target_h, target_w):
    """CLIPSeg를 이용해 텍스트에 해당하는 영역의 히트맵과 마스크 반환"""
    image = Image.open(image_path).convert("RGB")
    inputs = clipseg_processor(text=[text_prompt], images=[image], padding="max_length", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = clipseg_model(**inputs)
        preds = torch.sigmoid(outputs.logits)[0] # [352, 352]
    
    # Qwen Attention Map 크기로 리사이즈 (heatmap 계산용)
    heatmap_small = cv2.resize(preds.cpu().numpy(), (target_w, target_h))
    
    # Binary Mask (Refinement용)
    binary_mask = (heatmap_small > 0.15).astype(np.float32)
    return heatmap_small, binary_mask


def check_heatmap_containment(heatmap_top, heatmap_obj, threshold=0.15, containment_ratio=0.8):
    """
    Args:
        containment_ratio (float): Top 영역의 몇 % 이상이 Obj와 겹쳐야 포함으로 볼 것인지 (기본 0.9 = 90%)
    """
    
    # 1. 텐서인 경우 numpy 변환
    if hasattr(heatmap_top, 'cpu'):
        heatmap_top = heatmap_top.detach().cpu().numpy()
    if hasattr(heatmap_obj, 'cpu'):
        heatmap_obj = heatmap_obj.detach().cpu().numpy()

    # 2. 이진 마스크 생성
    mask_top = heatmap_top > threshold
    mask_obj = heatmap_obj > threshold

    # 3. 면적 계산
    area_top = np.sum(mask_top)
    area_obj = np.sum(mask_obj)

    # 예외 처리: Top 히트맵이 아예 활성화되지 않은 경우 (면적 0)
    if area_top == 0:
        return False

    # 조건 1: Top의 면적이 Object 면적보다 작은가?
    is_smaller = area_top < area_obj
    
    # 4. 포함 관계 확인 (수정된 부분)
    # 교집합(Intersection) 영역 계산
    intersection = np.logical_and(mask_top, mask_obj)
    intersection_area = np.sum(intersection)

    # [수정됨] 교집합 면적이 Top 전체 면적의 90% 이상인지 확인
    # (intersection_area / area_top) >= 0.9 와 동일한 수식입니다.
    is_inside = intersection_area >= (area_top * containment_ratio)

    # 디버깅용: 실제 겹치는 비율 확인
    # print(f"Overlap Ratio: {intersection_area / area_top:.2f}")

    return is_smaller and is_inside
# ------------------------------------------------------
# 3. 데이터셋 스캔
# ------------------------------------------------------
print(f"📂 {TARGET_ROOT} 디렉토리 스캔 중...")
data_list = []
for action in sorted(os.listdir(TARGET_ROOT)):
    action_path = os.path.join(TARGET_ROOT, action)
    if not os.path.isdir(action_path): continue
    for obj in sorted(os.listdir(action_path)):
        obj_path = os.path.join(action_path, obj)
        if not os.path.isdir(obj_path): continue
        for file in sorted(os.listdir(obj_path)):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                data_list.append({
                    'action': action,
                    'object': obj,
                    'filename': file,
                    'full_path': os.path.join(obj_path, file),
                    'gt_path': os.path.join(GT_ROOT, action, obj, file.replace('.jpg', '.png')) # GT 경로 추정
                })

df_fin = pd.DataFrame(data_list)
# 저장할 컬럼들 초기화
cols_to_add = ['output_sentence', 'top_token_text', 'following_text', 'clip_input', 'KLD', 'SIM', 'NSS']
for col in cols_to_add:
    df_fin[col] = None

print(f"✅ 총 {len(df_fin)}개의 데이터 준비 완료.")

# ------------------------------------------------------
# 4. Main Inference & Evaluation Loop
# ------------------------------------------------------
system_prompt = "You are a helpful language and vision assistant."

for index, row in tqdm(df_fin.iterrows(), total=len(df_fin), desc="Processing"):
    object_name = row['object'].replace("_", " ")
    action = row['action']
    file_path = row['full_path']
    gt_path = row['gt_path']
    filename = row['filename']
    
    # 원본 이미지 로드 (시각화용)
    orig_img = Image.open(file_path).convert("RGB")
    w, h = orig_img.size
    
    # --- [STEP 0] Qwen Inference ---
    description = f"When people perform {action} with {object_name}, which part of the {object_name} is used for '{action}'? answer in one sentence."
    image_base64 = make_input_image(file_path)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "image", "image": f"data:image/jpeg;base64,{image_base64}"}, {"type": "text", "text": description}]}
    ]
    
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False, use_cache=True)
        
    input_len = inputs.input_ids.shape[1]
    output_ids = generated_ids[0][input_len:]
    output_text = processor.decode(output_ids, skip_special_tokens=True)
    
    # --- [STEP 1] Attention Extraction ---
    with torch.no_grad():
        outputs = model(input_ids=generated_ids, pixel_values=inputs.pixel_values, image_grid_thw=inputs.image_grid_thw, attention_mask=torch.ones_like(generated_ids), output_attentions=True, return_dict=True)
    attentions = outputs.attentions
    
    # --- [STEP 2] Prepare Metadata ---
    ids_list = generated_ids[0].tolist()
    tok = processor.tokenizer
    
    if tok.convert_tokens_to_ids("<|vision_start|>") not in ids_list: continue
    vis_start_idx = ids_list.index(tok.convert_tokens_to_ids("<|vision_start|>"))
    vis_end_idx   = ids_list.index(tok.convert_tokens_to_ids("<|vision_end|>"))
    
    grid_t, grid_h, grid_w = inputs.image_grid_thw[0].detach().cpu().numpy()
    llm_grid_h, llm_grid_w = grid_h // 2, grid_w // 2 
    
    # --- [STEP 3] Base CLIPSeg (Object Name) ---
    clip_object_heatmap, clip_object_mask = get_clipseg_mask(file_path, object_name, llm_grid_h, llm_grid_w)

    # --- [STEP 4] Token Scoring & Selection ---
    token_candidates = []
    for q_idx in range(input_len, len(ids_list)):
        token_heatmap_accum = torch.zeros((llm_grid_h * llm_grid_w), device=device)
        for layer_attn in attentions:
            heads_attn = layer_attn[0, :, q_idx, vis_start_idx+1 : vis_end_idx]
            token_heatmap_accum += heads_attn.sum(dim=0)
        
        heatmap_np = token_heatmap_accum.reshape(llm_grid_h, llm_grid_w).cpu().numpy()
        s_img_masked = (heatmap_np * clip_object_mask).sum()
        
        token_candidates.append({
            "token_idx": q_idx,
            "token_str": tok.decode([ids_list[q_idx]]),
            "score": s_img_masked,
            "heatmap": heatmap_np.astype(np.float32) # 계산 정밀도 유지
        })
    
    if not token_candidates: continue
    
    # Sort & Select Top-1
    sorted_tokens = sorted(token_candidates, key=lambda x: x['score'], reverse=True)
    top_token = sorted_tokens[0]
    top_token_text = top_token['token_str']
    top_token_idx = top_token['token_idx'] - input_len
    
    # Find Following Token
    following_text = ""
    next_idx = top_token['token_idx'] + 1
    for cand in token_candidates:
        if cand['token_idx'] == next_idx:
            following_text = cand['token_str']
            break
            
    # --- [STEP 5] Refinement & Metrics (핵심 추가 부분) ---
    
    # 1. Specific CLIPSeg (Top Token + Next Token)
    # 구두점 제거 및 정리
    refined_prompt = f"{top_token_text} {following_text}".replace('.', '').strip()
    clip_specific_heatmap, clip_specific_mask = get_clipseg_mask(file_path, refined_prompt, llm_grid_h, llm_grid_w)
    
    # 2. Attention Map Normalization
    pos_map = top_token['heatmap'].copy()
    if pos_map.max() > 0: pos_map /= pos_map.max()
    
    # 3. Selection Logic (Containment Check)
    if check_heatmap_containment(clip_specific_mask, clip_object_mask):
        final_clip_heatmap = clip_specific_heatmap
        clipseg_input_text = refined_prompt
        print(f" -> Selected Specific Part: '{clipseg_input_text}'")
    else:
        final_clip_heatmap = clip_object_heatmap
        clipseg_input_text = object_name
        
    # 4. Hadamard Product & Blur
    # 사이즈 맞추기 (31x31 -> 31x31 그대로 연산)
    hadamard_map = pos_map * final_clip_heatmap
    hadamard_map = np.power(hadamard_map, 0.75) # Gamma Correction
    
    # 리사이즈 -> 블러 -> 정규화 (최종 Output 생성)
    final_map_resized = cv2.resize(hadamard_map, (w, h), interpolation=cv2.INTER_LINEAR)
    
    sig = min(w, h) * 0.05
    k_val = int(sig * 3) * 2 + 1
    blur_map = cv2.GaussianBlur(final_map_resized, (k_val, k_val), sig)
    final_result_map = min_max_normalize(blur_map)
    
    # 5. GT Evaluation
    gt_map = load_ground_truth(gt_path)
    metrics = {'KLD': 0.0, 'SIM': 0.0, 'NSS': 0.0}
    
    if gt_map is not None:
        metrics = calculate_metrics(final_result_map, gt_map)
        metrics_text = f"KLD:{metrics['KLD']:.2f} | SIM:{metrics['SIM']:.2f} | NSS:{metrics['NSS']:.2f}"
        
        metrics_tracker_alloutput.update(metrics) #
        metrics_tracker_alloutput.print_metrics(metrics, filename)

    else:
        metrics_text = "No GT"

    # --- [STEP 6] Save Results to DataFrame ---
    df_fin.at[index, 'output_sentence'] = output_text
    df_fin.at[index, 'top_token_text'] = top_token_text
    df_fin.at[index, 'following_text'] = following_text
    df_fin.at[index, 'clip_input'] = clipseg_input_text
    df_fin.at[index, 'KLD'] = metrics['KLD']
    df_fin.at[index, 'SIM'] = metrics['SIM']
    df_fin.at[index, 'NSS'] = metrics['NSS']
    
    # --- [STEP 7] Visualization ---
    fig, axes = plt.subplots(1, 6, figsize=(24, 5))
    
    # (1) Original
    axes[0].imshow(orig_img)
    axes[0].set_title(f"Original\n({object_name})")
    axes[0].axis('off')
    
    # (2) Raw Attention (Top-1)
    att_vis = cv2.resize(pos_map, (w, h), interpolation=cv2.INTER_NEAREST)
    axes[1].imshow(att_vis, cmap='jet')
    axes[1].set_title(f"Top-1 Attn\n('{top_token_text}')")
    axes[1].axis('off')
    
    # (3) Selected CLIPSeg
    clip_vis = cv2.resize(final_clip_heatmap, (w, h), interpolation=cv2.INTER_NEAREST)
    axes[2].imshow(clip_vis, cmap='viridis')
    axes[2].set_title(f"CLIPSeg Mask\n('{clipseg_input_text}')")
    axes[2].axis('off')
    
    # (4) Hadamard (Before Blur)
    hada_vis = cv2.resize(hadamard_map, (w, h), interpolation=cv2.INTER_NEAREST)
    axes[3].imshow(hada_vis, cmap='jet')
    axes[3].set_title("Hadamard\n(Attn x CLIP)")
    axes[3].axis('off')
    
    # (5) Final Result (Blurred)
    axes[4].imshow(final_result_map, cmap='jet')
    axes[4].set_title(f"Final Output\n{metrics_text}")
    axes[4].axis('off')
    
    # (6) GT
    if gt_map is not None:
        axes[5].imshow(gt_map, cmap='gray')
        axes[5].set_title("Ground Truth")
    else:
        axes[5].set_title("No GT")
    axes[5].axis('off')
    
    main_title = f"Obj: {object_name} | Act: {action} |{metrics_text}\nTop Tokens: [{top_token_text}({top_token_idx } ), clipseg input : {clipseg_input_text}] \n Whole answer : {output_text}"
    
    fig.suptitle(main_title, fontsize=12)
    
    save_path = os.path.join(VIS_DIR, f"{action}_{object_name}_{filename.split('.')[0]}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

    # --- [STEP 8] Memory Cleanup ---
    if index % 50 == 0:
        df_fin.to_pickle(SAVE_FILENAME)
    
    del generated_ids, outputs, attentions, token_candidates, sorted_tokens
    torch.cuda.empty_cache()
    gc.collect()

    # break


# 최종 저장
df_fin.to_pickle(SAVE_FILENAME)
print(f"🎉 모든 작업 완료! 결과 저장: {SAVE_FILENAME}")
print(f"📊 평균 메트릭 - KLD: {df_fin['KLD'].mean():.4f}, SIM: {df_fin['SIM'].mean():.4f}, NSS: {df_fin['NSS'].mean():.4f}")
