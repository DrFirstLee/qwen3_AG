# nohup python -u all_coocur_exo_test.py >> 2b_all_coocur_exo.log 2>&1 &

import os
import sys
import gc
import cv2
import json
import random
import numpy as np
import pandas as pd
import pickle
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, CLIPSegProcessor, CLIPSegForImageSegmentation

# --- 경로 설정 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("/home/bongo/porter_notebook/research/qwen3") 

from file_managing import (
    make_input_image_exo,
    calculate_metrics,
    load_ground_truth,
    prompt_dict_obj
)
from config import AGD20K_PATH, model_name, model_size

# ------------------------------------------------------
# 1. 환경 설정
# ------------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ENABLE_SDPA"] = "1"

TARGET_ROOT = f"{AGD20K_PATH}/Seen/testset/egocentric"
GT_ROOT = f"{AGD20K_PATH}/Seen/testset/GT"

EXO_CACHE_PATH = "fixed_exo_candidates.pkl"  # [변경] 미리 생성한 캐시 파일 경로
EXO_ROOT_BASE = Path(f"{AGD20K_PATH}/Seen/trainset/exocentric")

SAVE_PKL_NAME = f"{model_size}_all_coocur_exo_test.pkl"
VIS_ROOT = f"{model_size}_vis_all_coocur_exo_analysis"

os.makedirs(VIS_ROOT, exist_ok=True)

print(f"🤖 Loading Model: {model_name}")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map="cuda",
)
processor = AutoProcessor.from_pretrained(model_name)
device = model.device

print(f"👁️ Loading CLIPSeg...")
clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)


# # CLIP 모델 로드 (가볍고 빠른 ViT-B/32 또는 성능 좋은 ViT-L/14 사용)
# clip_model_id = "openai/clip-vit-large-patch14" 
# print(f"Loading CLIP: {clip_model_id}...")
# from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
# device = "cuda" if torch.cuda.is_available() else "cpu"
# clip_model = CLIPModel.from_pretrained(clip_model_id).to(device)
# clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
# clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_id)



# ------------------------------------------------------
# 2. Helper Functions (생략 없이 포함)
# ------------------------------------------------------
def min_max_normalize(map_data):
    m_min, m_max = map_data.min(), map_data.max()
    if m_max - m_min == 0: return map_data
    return (map_data - m_min) / (m_max - m_min)

def get_clipseg_mask(image_path, text_prompt, target_h, target_w):
    image = Image.open(image_path).convert("RGB")
    inputs = clipseg_processor(text=[text_prompt], images=[image], padding="max_length", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = clipseg_model(**inputs)
        preds = torch.sigmoid(outputs.logits)[0]
    heatmap_small = cv2.resize(preds.cpu().numpy(), (target_w, target_h))
    binary_mask = (heatmap_small > 0.15).astype(np.float32)
    return heatmap_small, binary_mask

def check_heatmap_containment(heatmap_top, heatmap_obj, threshold=0.15, containment_ratio=0.8):
    if hasattr(heatmap_top, 'cpu'): heatmap_top = heatmap_top.detach().cpu().numpy()
    if hasattr(heatmap_obj, 'cpu'): heatmap_obj = heatmap_obj.detach().cpu().numpy()
    mask_top = heatmap_top > threshold
    mask_obj = heatmap_obj > threshold
    area_top = np.sum(mask_top)
    if area_top == 0: return False
    is_smaller = area_top < np.sum(mask_obj)
    intersection = np.logical_and(mask_top, mask_obj)
    is_inside = np.sum(intersection) >= (area_top * containment_ratio)
    return is_smaller and is_inside

def apply_post_processing(heatmap, refinement_heatmap=None, w=224, h=224):
    if refinement_heatmap is not None:
        if heatmap.max() > 0: heatmap /= heatmap.max()
        heatmap = heatmap * refinement_heatmap
        heatmap = np.power(heatmap, 0.75)
    else:
        if heatmap.max() > 0: heatmap /= heatmap.max()
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    sig = min(w, h) * 0.05
    k_val = int(sig * 3) * 2 + 1
    blur_map = cv2.GaussianBlur(heatmap_resized, (k_val, k_val), sig)
    return min_max_normalize(blur_map)

def extract_exp5_map(attentions, ids_list, input_len, vis_start, vis_end, ego_path, object_name, llm_grid_h, llm_grid_w):
    device = attentions[0].device
    grid_h, grid_w = llm_grid_h, llm_grid_w
    
    token_candidates = []
    clip_obj_heatmap, clip_obj_mask = get_clipseg_mask(ego_path, object_name.replace('_', ' '), grid_h, grid_w)

    for q_idx in range(input_len, len(ids_list)):
        step_heatmap = torch.zeros((grid_h * grid_w), device=device)
        for layer_attn in attentions:
            heads_attn = layer_attn[0, :, q_idx, vis_start+1 : vis_end]
            step_heatmap += heads_attn.sum(dim=0)
        
        heatmap_np = step_heatmap.reshape(grid_h, grid_w).cpu().numpy().astype(np.float32)
        score = (heatmap_np * clip_obj_mask).sum()
        
        token_candidates.append({
            "idx": q_idx,
            "str": processor.tokenizer.decode([ids_list[q_idx]]),
            "score": score,
            "heatmap": heatmap_np
        })

    if token_candidates:
        sorted_cand = sorted(token_candidates, key=lambda x: x['score'], reverse=True)
        top_token = sorted_cand[0]
        map_top1 = top_token['heatmap']
        top_token_text = top_token['str']
        next_idx = top_token['idx'] + 1
        following_text = next((c['str'] for c in token_candidates if c['idx'] == next_idx), "")
    else:
        map_top1 = np.zeros((grid_h, grid_w), dtype=np.float32)
        top_token_text, following_text = "none", ""

    refined_prompt = f"{top_token_text} {following_text}".replace('.', '').strip()
    clip_spec_heatmap, clip_spec_mask = get_clipseg_mask(ego_path, refined_prompt, grid_h, grid_w)

    if check_heatmap_containment(clip_spec_mask, clip_obj_mask):
        adaptive_refine = clip_spec_heatmap
    else:
        adaptive_refine = clip_obj_heatmap

    # final_map = apply_post_processing(map_top1.copy(), refinement_heatmap=adaptive_refine)
    # [수정됨] w=w, h=h 를 반드시 전달해야 원본 크기로 복원됨
    final_map = apply_post_processing(map_top1.copy(), refinement_heatmap=adaptive_refine, w=w, h=h)
    
    return final_map, refined_prompt

# ------------------------------------------------------
# 3. 데이터 로딩
# ------------------------------------------------------

# [변경] Exo Cache 로드
if not os.path.exists(EXO_CACHE_PATH):
    print(f"❌ Error: Cache file {EXO_CACHE_PATH} not found. Run make_exo_cache.py first.")
    exit() # try-except 쓰지 말래서 그냥 exit

with open(EXO_CACHE_PATH, 'rb') as f:
    exo_cache_data = pickle.load(f)
print(f"✅ Loaded Exo Cache for {len(exo_cache_data)} cases.")

print(f"📂 {TARGET_ROOT} 디렉토리 스캔 중...")

target_samples = []

# Action -> Object -> File 구조 순회
for action in sorted(os.listdir(TARGET_ROOT)):
    action_path = os.path.join(TARGET_ROOT, action)
    if not os.path.isdir(action_path): continue

    for obj in sorted(os.listdir(action_path)):
        obj_path = os.path.join(action_path, obj)
        if not os.path.isdir(obj_path): continue

        for file in sorted(os.listdir(obj_path)):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                
                # Ego 이미지 경로
                ego_path = os.path.join(obj_path, file)
                
                # GT 이미지 경로 (확장자를 png로 변경)
                # os.path.join을 사용하여 안전하게 경로 생성
                gt_filename = os.path.splitext(file)[0] + ".png"
                gt_path = os.path.join(GT_ROOT, action, obj, gt_filename)

                target_samples.append({
                    "case_id": f"{action}_{obj}", # JSON의 key가 없으므로 파일명(확장자 제외)을 ID로 사용
                    "action": action,
                    "object": obj,
                    "ego_path": ego_path,
                    "gt_path": gt_path,
                    "filename": file
                })

# 결과 데이터프레임 생성
df_results = pd.DataFrame(target_samples)

print(f"✅ 총 {len(df_results)}개의 testset data를 로드했습니다.")
## 전체 exo 리스트 + 사람수 데이터 미리 불러오기

# df_person = pd.read_pickle("exo_person.pkl")
# df_person

# df_best = pd.read_pickle("/home/bongo/porter_notebook/research/qwen3/ECCV_codes/ablations/Seen/2B_all_trials_metrics.pkl")
# df_best

# ------------------------------------------------------
# 4. Main Loop & Accumulators
# ------------------------------------------------------
system_prompt = "You are a helpful language and vision assistant."
v_start_token = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
v_end_token = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

random_selection_df = pd.read_pickle("both_random_exo.pkl")
random_selection_df = random_selection_df[['case_id', 'action', 'object', 'exo_path']]

# [NEW] 누적 메트릭 저장을 위한 변수 초기화
cum_metrics = {
    'ego': {'KLD': 0.0, 'SIM': 0.0, 'NSS': 0.0},
    'random': {'KLD': 0.0, 'SIM': 0.0, 'NSS': 0.0},
    'no_coocur': {'KLD': 0.0, 'SIM': 0.0, 'NSS': 0.0},
    'coocur': {'KLD': 0.0, 'SIM': 0.0, 'NSS': 0.0},
}
valid_count = 0

all_trials_rows = []

for index, row in tqdm(df_results.iterrows(), total=len(df_results), desc="exo selection Analysis"):
    case_id = row['case_id']
    action = row['action']
    object_name = row['object']
    ego_path = row['ego_path']
    ego_filename = os.path.basename(ego_path)
    gt_path = row['gt_path']
    
    if not os.path.exists(gt_path): continue
        
    orig_img = Image.open(ego_path).convert("RGB")
    w, h = orig_img.size
    gt_map = load_ground_truth(gt_path)
    if gt_map.shape != (h, w):
        gt_map = cv2.resize(gt_map, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # --- [STEP 0] Ego-Only Baseline ---
    desc_ego = f"When people perform {action} with {object_name.replace('_',' ')}, which part of the {object_name.replace('_',' ')} is used for '{action}'? Answer in one sentence."
    ego_b64 = make_input_image_exo(ego_path)
    
    msg_ego = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [
            {"type": "image", "image": f"data:image/jpeg;base64,{ego_b64}"},
            {"type": "text", "text": desc_ego}
        ]}
    ]
    

    in_ego = processor.apply_chat_template(msg_ego, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(device)
    with torch.no_grad():
        gen_ego = model.generate(**in_ego, max_new_tokens=128, do_sample=False)
        out_ego = model(input_ids=gen_ego, pixel_values=in_ego.pixel_values, image_grid_thw=in_ego.image_grid_thw, attention_mask=torch.ones_like(gen_ego), output_attentions=True, return_dict=True)
        
    ids_ego = gen_ego[0].tolist()
    v_starts = [i for i, x in enumerate(ids_ego) if x == v_start_token]
    ego_s, ego_e = v_starts[0], [i for i, x in enumerate(ids_ego) if x == v_end_token][0]
    
    grid_t, grid_h, grid_w = in_ego.image_grid_thw[0].detach().cpu().numpy()
    
    map_ego, _ = extract_exp5_map(out_ego.attentions, ids_ego, in_ego.input_ids.shape[1], ego_s, ego_e, ego_path, object_name, grid_h//2, grid_w//2)
    metrics_ego = calculate_metrics(map_ego, gt_map)
        
   
    desc_context = f"Refer to the second image (exocentric view) for context. Based on the first image (egocentric view), when people perform {action} with {object_name.replace('_',' ')}, which part of the {object_name.replace('_',' ')} is used for '{action}'? Answer in one sentence."

    exo_file = random_selection_df[random_selection_df['case_id'] == case_id]['exo_path'].values[0]
    exo_file = exo_file.replace("/home/DATA/AGD20K", AGD20K_PATH )
    exo_file_random = exo_file
    exo_b64 = make_input_image_exo(str(exo_file))
    msg_context = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "image", "image": f"data:image/jpeg;base64,{ego_b64}"},
                {"type": "image", "image": f"data:image/jpeg;base64,{exo_b64}"},
                {"type": "text", "text": desc_context}
            ]}
        ]
        
    in_ctx = processor.apply_chat_template(msg_context, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(device)
    with torch.no_grad():
        gen_ctx = model.generate(**in_ctx, max_new_tokens=128, do_sample=False)
        out_ctx = model(input_ids=gen_ctx, pixel_values=in_ctx.pixel_values, image_grid_thw=in_ctx.image_grid_thw, attention_mask=torch.ones_like(gen_ctx), output_attentions=True, return_dict=True)
    
    ids_ctx = gen_ctx[0].tolist()
    v_starts = [i for i, x in enumerate(ids_ctx) if x == v_start_token]
    ego_s, ego_e = v_starts[0], [i for i, x in enumerate(ids_ctx) if x == v_end_token][0]
    
    grid_t, grid_h, grid_w = in_ctx.image_grid_thw[0].detach().cpu().numpy()
    
    map_exo, refined_prompt = extract_exp5_map(out_ctx.attentions, ids_ctx, in_ctx.input_ids.shape[1], ego_s, ego_e, ego_path, object_name, grid_h//2, grid_w//2)
    metrics_exo = calculate_metrics(map_exo, gt_map)

    ## 여기다가 NO CO OCCUR
    df_no_coocur = pd.read_pickle("/home/bongo/porter_notebook/research/qwen3/ECCV_codes/ablations/Seen/unique_first_exo_candidates.pkl")
    no_coocur_exo_file = df_no_coocur[case_id][0]
    no_coocur_exo_file = no_coocur_exo_file.replace("/home/DATA/AGD20K", AGD20K_PATH )

    no_coocur_exo_b64 = make_input_image_exo(str(no_coocur_exo_file))

    msg_context = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "image", "image": f"data:image/jpeg;base64,{ego_b64}"},
                {"type": "image", "image": f"data:image/jpeg;base64,{no_coocur_exo_b64}"},
                {"type": "text", "text": desc_context}
            ]}
        ]
        
    in_ctx = processor.apply_chat_template(msg_context, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(device)
    with torch.no_grad():
        gen_ctx = model.generate(**in_ctx, max_new_tokens=128, do_sample=False)
        out_ctx = model(input_ids=gen_ctx, pixel_values=in_ctx.pixel_values, image_grid_thw=in_ctx.image_grid_thw, attention_mask=torch.ones_like(gen_ctx), output_attentions=True, return_dict=True)
    
    ids_ctx = gen_ctx[0].tolist()
    v_starts = [i for i, x in enumerate(ids_ctx) if x == v_start_token]
    ego_s, ego_e = v_starts[0], [i for i, x in enumerate(ids_ctx) if x == v_end_token][0]
    
    grid_t, grid_h, grid_w = in_ctx.image_grid_thw[0].detach().cpu().numpy()
    
    map_no_coocur, refined_prompt = extract_exp5_map(out_ctx.attentions, ids_ctx, in_ctx.input_ids.shape[1], ego_s, ego_e, ego_path, object_name, grid_h//2, grid_w//2)
    metrics_no_coocur = calculate_metrics(map_no_coocur, gt_map)


    ## 여기다가  CO OCCUR
    df_coocur = pd.read_pickle("/home/bongo/porter_notebook/research/qwen3/ECCV_codes/ablations/Seen/overlap_first_exo_candidates.pkl")
    coocur_exo_file = df_coocur[case_id][0]
    coocur_exo_file = coocur_exo_file.replace("/home/DATA/AGD20K", AGD20K_PATH )

    coocur_exo_b64 = make_input_image_exo(str(coocur_exo_file))

    msg_context = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "image", "image": f"data:image/jpeg;base64,{ego_b64}"},
                {"type": "image", "image": f"data:image/jpeg;base64,{coocur_exo_b64}"},
                {"type": "text", "text": desc_context}
            ]}
        ]
        
    in_ctx = processor.apply_chat_template(msg_context, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(device)
    with torch.no_grad():
        gen_ctx = model.generate(**in_ctx, max_new_tokens=128, do_sample=False)
        out_ctx = model(input_ids=gen_ctx, pixel_values=in_ctx.pixel_values, image_grid_thw=in_ctx.image_grid_thw, attention_mask=torch.ones_like(gen_ctx), output_attentions=True, return_dict=True)
    
    ids_ctx = gen_ctx[0].tolist()
    v_starts = [i for i, x in enumerate(ids_ctx) if x == v_start_token]
    ego_s, ego_e = v_starts[0], [i for i, x in enumerate(ids_ctx) if x == v_end_token][0]
    
    grid_t, grid_h, grid_w = in_ctx.image_grid_thw[0].detach().cpu().numpy()
    
    map_coocur, refined_prompt = extract_exp5_map(out_ctx.attentions, ids_ctx, in_ctx.input_ids.shape[1], ego_s, ego_e, ego_path, object_name, grid_h//2, grid_w//2)
    metrics_coocur = calculate_metrics(map_coocur, gt_map)


    # 1. 누적 업데이트
    valid_count += 1
    for k in ['KLD', 'SIM', 'NSS']:
        # Ego
        if not np.isnan(metrics_ego[k]):
            cum_metrics['ego'][k] += metrics_ego[k]
        
        # Random
        if not np.isnan(metrics_exo[k]):
            cum_metrics['random'][k] += metrics_exo[k]
        
        # NO coocur
        if not np.isnan(metrics_no_coocur[k]):
            cum_metrics['no_coocur'][k] += metrics_no_coocur[k]

        # CO coocur
        if not np.isnan(metrics_coocur[k]):
            cum_metrics['coocur'][k] += metrics_coocur[k]

    # 2. DataFrame 저장
    df_results.at[index, 'ego_kld'] = metrics_ego['KLD']
    df_results.at[index, 'random'] = metrics_exo['KLD']
    df_results.at[index, 'random_exo'] = exo_file_random
    df_results.at[index, 'no_coocur'] = metrics_no_coocur['KLD']
    df_results.at[index, 'no_coocur_exo'] = no_coocur_exo_file
    df_results.at[index, 'coocur'] = metrics_coocur['KLD']
    df_results.at[index, 'coocur_exo'] = coocur_exo_file

    # ------------------------------------------------------
    # [VISUALIZATION] 3x2 Grid Result Saving
    # ------------------------------------------------------
    # 1. 이미지 준비 (RGB 변환 및 리사이즈)
    ego_img_np = np.array(orig_img) # RGB
    h, w = ego_img_np.shape[:2]

    # Exo 이미지 로드 함수
    def load_and_resize_exo(path, target_h, target_w):
        if os.path.exists(path):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (target_w, target_h))
            return img
        else:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # no coocur 로드
    rand_exo_img = load_and_resize_exo(exo_file_random, h, w)
    no_coocur_exo_img = load_and_resize_exo(no_coocur_exo_file, h, w)
    coocur_exo_img = load_and_resize_exo(coocur_exo_file, h, w)

    # 2. 히트맵 오버레이 함수 (JET Colormap 적용)
    def create_overlay(base_img, heatmap):
        # Heatmap 정규화 (0~255)
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)
        
        # ColorMap 적용
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # 오버레이 (원본 60% + 히트맵 40%)
        return cv2.addWeighted(base_img, 0.6, heatmap_color, 0.4, 0)

    # 각 결과에 대한 오버레이 생성
    overlay_ego = create_overlay(ego_img_np, map_ego)
    overlay_rand = create_overlay(ego_img_np, map_exo)
    overlay_no_coocur = create_overlay(ego_img_np, map_no_coocur)
    overlay_coocur = create_overlay(ego_img_np, map_coocur)

    # 3. Matplotlib Plotting (3x2 Grid -> 4x2 Grid)
    fig, axes = plt.subplots(2, 5, figsize=(24, 10))
    
    # --- Row 1: Source Images & GT Mask ---
    # (1,1) Ego Image
    axes[0, 0].imshow(ego_img_np)
    axes[0, 0].set_title(f"Ego Input\n({case_id})", fontsize=12)
    axes[0, 0].axis('off')

    # (1,2) Random Exo Image
    axes[0, 1].imshow(rand_exo_img)
    axes[0, 1].set_title("Random Exo Input", fontsize=12)
    axes[0, 1].axis('off')

    # (1,3) No Co-occurrence Exo Image
    axes[0, 2].imshow(no_coocur_exo_img)
    axes[0, 2].set_title(f"No Co-occurrence Exo Input", fontsize=12)
    axes[0, 2].axis('off')

    # (1,4) Co-occurrence Exo Image
    axes[0, 3].imshow(coocur_exo_img)
    axes[0, 3].set_title(f"Co-occurrence Exo Input", fontsize=12)
    axes[0, 3].axis('off')

    # (1,5) Ground Truth (Binary Mask)
    axes[0, 4].imshow(gt_map, cmap='gray')
    axes[0, 4].set_title("Ground Truth Mask", fontsize=12)
    axes[0, 4].axis('off')

    # --- Row 2: Result Heatmaps (Overlay) ---
    # (2,1) Ego-Only Result
    axes[1, 0].imshow(overlay_ego)
    axes[1, 0].set_title(f"Ego-Only Result\nKLD: {metrics_ego['KLD']:.4f}", fontsize=12, fontweight='bold', color='blue')
    axes[1, 0].axis('off')

    # (2,2) Random Exo Result
    axes[1, 1].imshow(overlay_rand)
    axes[1, 1].set_title(f"Random Exo Context\nKLD: {metrics_exo['KLD']:.4f}", fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    # (2,3) No Co-occurrence Exo Result
    axes[1, 2].imshow(overlay_no_coocur)
    # 성능 향상 여부에 따라 색상 다르게 표시 (개선되면 빨간색)
    title_color = 'red' if metrics_no_coocur['KLD'] < metrics_exo['KLD'] else 'black'
    axes[1, 2].set_title(f"No Co-occurrence Exo Context\nKLD: {metrics_no_coocur['KLD']:.4f}", fontsize=12, fontweight='bold', color=title_color)
    axes[1, 2].axis('off')

    # (2,4) Co-occurrence Exo Result
    axes[1, 3].imshow(overlay_coocur)
    # 성능 향상 여부에 따라 색상 다르게 표시 (개선되면 빨간색)
    title_color = 'red' if metrics_coocur['KLD'] < metrics_exo['KLD'] else 'black'
    axes[1, 3].set_title(f"Co-occurrence Exo Context\nKLD: {metrics_coocur['KLD']:.4f}", fontsize=12, fontweight='bold', color=title_color)
    axes[1, 3].axis('off')

    # (2,5) Empty
    axes[1, 4].axis('off')
    
    # Layout 조정 및 저장
    plt.tight_layout()
    save_path = os.path.join(VIS_ROOT, f"{case_id}_{ego_filename}.png")
    plt.savefig(save_path, dpi=100)
    plt.close(fig) # 메모리 누수 방지


    # 4. 실시간 누적 평균 출력 (동일)
    avg_ego = {k: v/valid_count for k, v in cum_metrics['ego'].items()}
    avg_random = {k: v/valid_count for k, v in cum_metrics['random'].items()}
    avg_no_coocur = {k: v/valid_count for k, v in cum_metrics['no_coocur'].items()}
    avg_coocur = {k: v/valid_count for k, v in cum_metrics['coocur'].items()}   
    
    print(f"\n📊 [Avg Metrics @ {valid_count}]")
    print(f"   Baseline (Ego)   : KLD {avg_ego['KLD']:.3f} | SIM {avg_ego['SIM']:.3f} | NSS {avg_ego['NSS']:.3f}")
    print(f"   Random  : KLD {avg_random['KLD']:.3f} | SIM {avg_random['SIM']:.3f} | NSS {avg_random['NSS']:.3f}")
    print(f"   No Co-occurrence : KLD {avg_no_coocur['KLD']:.3f} | SIM {avg_no_coocur['SIM']:.3f} | NSS {avg_no_coocur['NSS']:.3f}")
    print(f"   Co-occurrence    : KLD {avg_coocur['KLD']:.3f} | SIM {avg_coocur['SIM']:.3f} | NSS {avg_coocur['NSS']:.3f}")
    print("-" * 60)

    if index % 5 == 0:
        df_results.to_pickle(SAVE_PKL_NAME)

df_results.to_pickle(SAVE_PKL_NAME)
print("\n🎉 Analysis Complete!")