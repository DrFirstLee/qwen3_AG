#  nohup python -u seen_every_case.py >> 2B_seen_some_case.log 2>&1 & 
#  nohup python -u seen_every_case.py >> 2B_seen_all_case.log 2>&1 & 
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
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, CLIPSegProcessor, CLIPSegForImageSegmentation

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append("/home/bongo/porter_notebook/research/qwen3")
from file_managing import (
    make_input_image,
    calculate_metrics, 
    load_ground_truth ,
    prompt_dict_obj
)

# ★ MetricsTracker 임포트
from VLM_model_dot_relative import MetricsTracker

from config import AGD20K_PATH, model_name

# --- [추가] Exo 이미지 로드 함수 ---
def make_input_image_exo(path):
    # 기존 make_input_image와 동일하거나 exo 전용 전처리가 필요할 경우 수정
    return make_input_image(path)

# ------------------------------------------------------
# 1. 환경 설정 및 모델 로딩
# ------------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ENABLE_SDPA"] = "1"

TARGET_ROOT = f"{AGD20K_PATH}/Seen/testset/egocentric"
GT_ROOT = f"{AGD20K_PATH}/Seen/testset/GT"

# 이전 단계에서 저장한 최적의 Exo 이미지 DB 로드
EXO_DB_PATH = "selected_best_exo_images.pkl" 
if not os.path.exists(EXO_DB_PATH):
    raise FileNotFoundError(f"Exo DB를 먼저 생성해야 합니다: {EXO_DB_PATH}")
df_exo_db = pd.read_pickle(EXO_DB_PATH)


SAVE_FILENAME = "2B_attention_result_seen_all.pkl"
VIS_DIR = "2B_result_vis_all"
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
# 2. Metrics Trackers 초기화 (실험별 객체 생성)
# ------------------------------------------------------
# 5가지 실험에 대한 트래커 생성
# (Experiment 리스트 및 트래커 초기화 부분은 기존과 동일)
trackers = {}
# Context (Exo+Ego) 실험: Exp1 ~ Exp6
for i in range(1, 7):
    trackers[f'Exp{i}'] = MetricsTracker(name=f"Exp{i}_Context")
# Baseline (Ego Only) 실험: Exp7 ~ Exp12
for i in range(7, 13):
    trackers[f'Exp{i}'] = MetricsTracker(name=f"Exp{i}_EgoOnly")
# ------------------------------------------------------
# 3. Helper Functions
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

def extract_all_maps(attentions, ids_list, input_len, vis_start, vis_end, ego_path, object_name, plsp_name,llm_grid_h, llm_grid_w ):
    """
    attentions: 모델에서 출력된 attention 튜플
    ids_list: generated_ids 리스트
    input_len: prompt 입력 길이 (output 시작점)
    vis_start, vis_end: Grounding 대상이 되는 이미지(Ego)의 토큰 범위
    """
    # 0. 초기화
    device = attentions[0].device
    grid_h, grid_w = llm_grid_h, llm_grid_w  # 전역 변수 혹은 인자로 전달
    final_maps = {}
    
    # Ego 이미지 정보 (Post-processing용)
    orig_img = Image.open(ego_path).convert("RGB")
    w, h = orig_img.size

    # -------------------------------------------------------------------------
    # 1. Base Attention 추출 (Last Input, Avg Output, Top-1)
    # -------------------------------------------------------------------------
    
    # (1) Exp1: Last Input (질문 직후 Ego를 보는 눈)
    last_input_idx = input_len - 1
    map_last_input = torch.zeros((grid_h * grid_w), device=device)
    for layer_attn in attentions:
        # last_input_idx 토큰이 vis_start:vis_end 영역을 보는 어텐션
        heads_attn = layer_attn[0, :, last_input_idx, vis_start+1 : vis_end]
        map_last_input += heads_attn.sum(dim=0)
    np_map_last_input = map_last_input.reshape(grid_h, grid_w).cpu().numpy().astype(np.float32)

    # (2) Exp2 & Top-1 스코어링
    map_avg_accum = torch.zeros((grid_h * grid_w), device=device)
    token_candidates = []
    token_count = 0
    
    # Ego Object Mask (Top-1 선정을 위한 필터)
    clip_obj_heatmap, clip_obj_mask = get_clipseg_mask(ego_path, object_name.replace('_', ' '), grid_h, grid_w)

    for q_idx in range(input_len, len(ids_list)):
        step_heatmap = torch.zeros((grid_h * grid_w), device=device)
        for layer_attn in attentions:
            heads_attn = layer_attn[0, :, q_idx, vis_start+1 : vis_end]
            step_heatmap += heads_attn.sum(dim=0)
        
        map_avg_accum += step_heatmap
        token_count += 1
        
        heatmap_np = step_heatmap.reshape(grid_h, grid_w).cpu().numpy().astype(np.float32)
        # 해당 토큰이 Ego Object 영역을 얼마나 설명하는지 점수화
        score = (heatmap_np * clip_obj_mask).sum()
        
        token_candidates.append({
            "idx": q_idx,
            "str": processor.tokenizer.decode([ids_list[q_idx]]),
            "score": score,
            "heatmap": heatmap_np
        })

    # Exp2: Avg Output
    if token_count > 0:
        map_avg_output = (map_avg_accum / token_count).reshape(grid_h, grid_w).cpu().numpy().astype(np.float32)
    else:
        map_avg_output = np.zeros((grid_h, grid_w), dtype=np.float32)

    # Exp3: Top-1 (Ego-best token)
    if token_candidates:
        sorted_cand = sorted(token_candidates, key=lambda x: x['score'], reverse=True)
        top_token = sorted_cand[0]
        map_top1 = top_token['heatmap']
        top_token_text = top_token['str']
        
        # Following text (for Exp5 context)
        next_idx = top_token['idx'] + 1
        following_text = next((c['str'] for c in token_candidates if c['idx'] == next_idx), "")
    else:
        map_top1 = np.zeros((grid_h, grid_w), dtype=np.float32)
        top_token_text, following_text = "none", ""

    # -------------------------------------------------------------------------
    # 2. 가공 및 Refinement (Exp 1 ~ 6)
    # -------------------------------------------------------------------------
    
    # Exp 1, 2, 3 (Raw Attention Maps)
    final_maps['Exp1'] = apply_post_processing(np_map_last_input.copy(), w=w, h=h)
    final_maps['Exp2'] = apply_post_processing(map_avg_output.copy(), w=w, h=h)
    final_maps['Exp3'] = apply_post_processing(map_top1.copy(), w=w, h=h)
    
    # Exp 4: Top-1 + Ego Object Prior
    final_maps['Exp4'] = apply_post_processing(map_top1.copy(), refinement_heatmap=clip_obj_heatmap, w=w, h=h)
    
    # Exp 5: Top-1 + Adaptive Part/Obj Prior
    refined_prompt = f"{top_token_text} {following_text}".replace('.', '').strip()
    clip_spec_heatmap, clip_spec_mask = get_clipseg_mask(ego_path, refined_prompt, grid_h, grid_w)
    
    if check_heatmap_containment(clip_spec_mask, clip_obj_mask):
        adaptive_refine = clip_spec_heatmap # Part로 인식됨
    else:
        adaptive_refine = clip_obj_heatmap # Obj로 Fallback
    final_maps['Exp5'] = apply_post_processing(map_top1.copy(), refinement_heatmap=adaptive_refine, w=w, h=h)
    
    # Exp 6: Top-1 + PLSP (Semantic Prior)
    clip_plsp_heatmap, _ = get_clipseg_mask(ego_path, plsp_name.replace('_', ' '), grid_h, grid_w)
    final_maps['Exp6'] = apply_post_processing(map_top1.copy(), refinement_heatmap=clip_plsp_heatmap, w=w, h=h)

    return final_maps, refined_prompt
    
# ------------------------------------------------------
# 4. 데이터셋 스캔
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
                    'gt_path': os.path.join(GT_ROOT, action, obj, file.replace('.jpg', '.png'))
                })

df_raw = pd.DataFrame(data_list)

# --- [추가] Action-Object 페어별로 1개만 샘플링 ---
# 각 그룹(action, object)에서 첫 번째 데이터만 선택
# df_fin = df_raw.groupby(['action', 'object']).first().reset_index()

# 만약 무작위로 1개를 뽑고 싶다면 아래 코드를 사용하세요:
# df_fin = df_raw.groupby(['action', 'object']).sample(n=1, random_state=42).reset_index()
df_fin = df_raw.copy()
# 실험별 메트릭 컬럼 초기화
exp_names = [
    'Exp1_Context_LastInput', 'Exp2_Context_AvgOutput', 'Exp3_Context_Top1_Raw', 
    'Exp4_Context_Top1_Obj', 'Exp5_Context_Top1_Adapt', 'Exp6_Context_PLSP',
    'Exp7_EgoOnly_LastInput', 'Exp8_EgoOnly_AvgOutput', 'Exp9_EgoOnly_Top1_Raw',
    'Exp10_EgoOnly_Top1_Obj', 'Exp11_EgoOnly_Top1_Adapt', 'Exp12_EgoOnly_PLSP'
]
metrics_keys = ['KLD', 'SIM', 'NSS']
for exp in exp_names:
    for metric in metrics_keys:
        df_fin[f"{exp}_{metric}"] = None

df_fin['top_token_text'] = None 
df_fin['exo_filename'] = None # Exo 파일명 추적용

print(f"✅ 총 {len(df_fin)}개의 Action-Object 페어 준비 완료 (페어당 1개 샘플링).")

# ------------------------------------------------------
# 5. Main Loop
# ------------------------------------------------------
system_prompt = "You are a helpful language and vision assistant."

for index, row in tqdm(df_fin.iterrows(), total=len(df_fin), desc="Processing"):
    action = row['action']
    object_name = row['object']
    ego_path = row['full_path']
    gt_path = row['gt_path']  
    PLSP_name = prompt_dict_obj[action][row['object']]

    filename = row['filename']
    orig_img = Image.open(ego_path).convert("RGB")
    w, h = orig_img.size

    # --- [STEP 0] Exo Context 이미지 찾기 ---
    exo_row = df_exo_db[(df_exo_db['action'] == action) & (df_exo_db['object'] == object_name)]
    if exo_row.empty:
        # DB에 없으면 가장 첫 번째 이미지라도 가져오거나 스킵
        continue
    exo_path = exo_row.iloc[0]['best_exo_path']    
    v_start_token = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    v_end_token = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

    # -------------------------------------------------------------------------
    # PART A: Context-Aware Inference (Ego + Exo) -> Exp 1-6
    # -------------------------------------------------------------------------
    desc_context = f"Refer to the second image (exocentric view) for context. Based on the first image (egocentric view), when people perform {action} with {object_name.replace('_',' ')}, which part of the {object_name.replace('_',' ')} is used for '{action}'? Answer in one sentence."
    
    ego_b64 = make_input_image(ego_path)
    exo_b64 = make_input_image_exo(exo_path)

    msg_context = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [
            {"type": "image", "image": f"data:image/jpeg;base64,{ego_b64}"}, # Ego
            {"type": "image", "image": f"data:image/jpeg;base64,{exo_b64}"}, # Exo
            {"type": "text", "text": desc_context}
        ]}
    ]
    
    # --- [STEP A-1] Inference & Attention Extraction ---
    in_ctx = processor.apply_chat_template(msg_context, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(device)
    with torch.no_grad():
        gen_ctx = model.generate(**in_ctx, max_new_tokens=1024, do_sample=False)
        out_ctx = model(input_ids=gen_ctx, pixel_values=in_ctx.pixel_values, image_grid_thw=in_ctx.image_grid_thw, attention_mask=torch.ones_like(gen_ctx), output_attentions=True, return_dict=True)
    
    # 첫 번째 이미지(Ego) 영역 인덱스 찾기
    ids_ctx = gen_ctx[0].tolist()
    v_starts = [i for i, x in enumerate(ids_ctx) if x == v_start_token]; v_ends = [i for i, x in enumerate(ids_ctx) if x == v_end_token]
    ego_start, ego_end = v_starts[0], v_ends[0] # 첫 번째가 Ego

    grid_t, grid_h, grid_w = in_ctx.image_grid_thw[0].detach().cpu().numpy()
    llm_grid_h, llm_grid_w = grid_h // 2, grid_w // 2
    # 어텐션 계산 함수 호출 (Part A용)
    maps_context, adaptive_refine_exo = extract_all_maps(out_ctx.attentions, ids_ctx, in_ctx.input_ids.shape[1], ego_start, ego_end, ego_path, object_name, PLSP_name,llm_grid_h, llm_grid_w )

    # -------------------------------------------------------------------------
    # PART B: Ego-Only Inference (Ego) -> Exp 7-12
    # -------------------------------------------------------------------------
    desc_ego = f"When people perform {action} with {object_name.replace('_',' ')}, which part of the {object_name.replace('_',' ')} is used for '{action}'? Answer in one sentence."
    
    msg_ego = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [
            {"type": "image", "image": f"data:image/jpeg;base64,{ego_b64}"},
            {"type": "text", "text": desc_ego}
        ]}
    ]
    
    # --- [STEP B-1] Inference & Attention Extraction ---
    in_ego = processor.apply_chat_template(msg_ego, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(device)
    with torch.no_grad():
        gen_ego = model.generate(**in_ego, max_new_tokens=1024, do_sample=False)
        out_ego = model(input_ids=gen_ego, pixel_values=in_ego.pixel_values, image_grid_thw=in_ego.image_grid_thw, attention_mask=torch.ones_like(gen_ego), output_attentions=True, return_dict=True)
    
    # Ego 이미지 영역 인덱스 (이미지가 하나뿐이므로 index 0 사용)
    ids_ego = gen_ego[0].tolist()
    v_starts_e = [i for i, x in enumerate(ids_ego) if x == v_start_token]; v_ends_e = [i for i, x in enumerate(ids_ego) if x == v_end_token]
    ego_only_start, ego_only_end = v_starts_e[0], v_ends_e[0]
    
    # 어텐션 계산 함수 호출 (Part B용)
    maps_ego_only, adaptive_refine_ego = extract_all_maps(out_ego.attentions, ids_ego, in_ego.input_ids.shape[1], ego_only_start, ego_only_end, ego_path, object_name, PLSP_name, llm_grid_h, llm_grid_w)



    # -------------------------------------------------------------------------
    # [EVALUATION] GT 비교 및 12개 트래커 업데이트
    # -------------------------------------------------------------------------
    gt_map = load_ground_truth(gt_path)
    
    if gt_map is not None:
        if gt_map.shape != (h, w):
            gt_map = cv2.resize(gt_map, (w, h), interpolation=cv2.INTER_NEAREST)

        # 실험 매핑 정의
        # trackers['Exp1'] ~ ['Exp6'] : Context (Ego+Exo)
        # trackers['Exp7'] ~ ['Exp12'] : Ego-Only (Baseline)
        
        print(f"\n--- Metrics Update [{index}] ---")
        
        # 1. Context Experiments (1-6)
        for i in range(1, 7):
            exp_key = f'Exp{i}'
            pred_map = maps_context[exp_key]
            metrics = calculate_metrics(pred_map, gt_map)
            
            # Tracker & DataFrame 업데이트
            trackers[exp_key].update(metrics)
            for m_key in ['KLD', 'SIM', 'NSS']:
                df_fin.at[index, f"{exp_key}_Context_{m_key}"] = metrics[m_key]
            
            # 디버그 출력 (Exp3만 대표로)
            trackers[exp_key].print_metrics(metrics, f"[Ctx] {filename}")

        # 2. Ego-Only Experiments (7-12)
        for i in range(7, 13):
            exp_key = f'Exp{i}'
            # extract_all_maps는 내부적으로 Exp1~6 키를 사용하므로 i-6으로 접근
            pred_map = maps_ego_only[f'Exp{i-6}'] 
            metrics = calculate_metrics(pred_map, gt_map)
            
            # Tracker & DataFrame 업데이트
            trackers[exp_key].update(metrics)
            for m_key in ['KLD', 'SIM', 'NSS']:
                df_fin.at[index, f"{exp_key}_EgoOnly_{m_key}"] = metrics[m_key]

            trackers[exp_key].print_metrics(metrics, f"[Ego] {filename}")
        print(f"selected token exo :{adaptive_refine_exo} ego : {adaptive_refine_ego}")
# -------------------------------------------------------------------------
    # [VISUALIZATION] 3-Metric Display & Exo Filename Version
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 8, figsize=(32, 12)) # 텍스트 공간 확보를 위해 높이를 12로 조정
    
    # --- 공통 원본 및 GT 배치 ---
    axes[0, 0].imshow(orig_img)
    axes[0, 0].set_title(f"Ego Image\n{action}_{object_name}", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(orig_img)
    axes[1, 0].axis('off')

    # --- Exo Context 이미지 및 파일명 표시 ---
    exo_img_plot = Image.open(exo_path).convert("RGB")
    exo_filename = os.path.basename(exo_path) # 경로에서 파일명만 추출
    axes[0, 1].imshow(exo_img_plot)
    # 파일명이 길 경우를 대비해 폰트 크기를 조절하고 줄바꿈 적용 가능
    axes[0, 1].set_title(f"Exo Context\n({exo_filename})", fontsize=10, color='darkgreen')
    axes[0, 1].axis('off')

    # --- GT 배치 ---
    if gt_map is not None:
        axes[1, 1].imshow(gt_map, cmap='gray')
        axes[1, 1].set_title("Ground Truth", fontsize=12)
        axes[1, 1].axis('off')
    else:
        axes[1, 1].axis('off')

    # --- 실험 결과 배치 (Exp 1-6 & 7-12) ---
    titles = ["LastIn", "AvgOut", "Top1", "Top1+Obj", f"Top1+Adapt_exo:{adaptive_refine_exo}_ego:{adaptive_refine_ego}", f"Top1+PLSP({PLSP_name})"]
    for j in range(6):
        # 1. Context Row (Top)
        ax_ctx = axes[0, j+2]
        key_ctx = f'Exp{j+1}'
        ax_ctx.imshow(maps_context[key_ctx], cmap='jet')
        ax_ctx.set_title(f"Ctx_{titles[j]}", fontsize=11, fontweight='bold')
        ax_ctx.axis('off')
        
        # 2. Ego-Only Row (Bottom)
        ax_ego = axes[1, j+2]
        key_ego = f'Exp{j+7}'
        ax_ego.imshow(maps_ego_only[f'Exp{j+1}'], cmap='jet')
        ax_ego.set_title(f"Ego_{titles[j]}", fontsize=11, fontweight='bold')
        ax_ego.axis('off')
        
        # 3. 메트릭 표시 (KLD, SIM, NSS)
        if gt_map is not None:
            # 데이터프레임에서 값 가져오기
            k_c = df_fin.at[index, f"{key_ctx}_Context_KLD"]
            s_c = df_fin.at[index, f"{key_ctx}_Context_SIM"]
            n_c = df_fin.at[index, f"{key_ctx}_Context_NSS"]
            
            k_e = df_fin.at[index, f"{key_ego}_EgoOnly_KLD"]
            s_e = df_fin.at[index, f"{key_ego}_EgoOnly_SIM"]
            n_e = df_fin.at[index, f"{key_ego}_EgoOnly_NSS"]

            # 메트릭 텍스트 구성
            metric_text_ctx = f"K: {k_c:.2f}\nS: {s_c:.2f}\nN: {n_c:.2f}"
            metric_text_ego = f"K: {k_e:.2f}\nS: {s_e:.2f}\nN: {n_e:.2f}"

            # 텍스트 박스 배치
            ax_ctx.text(0.5, -0.05, metric_text_ctx, transform=ax_ctx.transAxes, 
                        ha='center', va='top', fontsize=10, color='blue', fontweight='semibold',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray', boxstyle='round,pad=0.3'))
            
            ax_ego.text(0.5, -0.05, metric_text_ego, transform=ax_ego.transAxes, 
                        ha='center', va='top', fontsize=10, color='red', fontweight='semibold',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray', boxstyle='round,pad=0.3'))

    # 레이아웃 조정
    plt.tight_layout()
    # 텍스트 박스가 하단 프레임에 가려지지 않도록 충분한 여백 확보
    plt.subplots_adjust(bottom=0.18, hspace=0.4) 
    
    # 결과 저장
    save_name = f"{action}_{object_name}_{filename.split('.')[0]}.png"
    save_path = os.path.join(VIS_DIR, save_name)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # [CLEANUP] Memory & Intermediate Saving
    # -------------------------------------------------------------------------
    if index % 50 == 0:
        df_fin.to_pickle(SAVE_FILENAME)
        print(f"💾 중간 저장 완료: {SAVE_FILENAME} ({index}/{len(df_fin)})")
    
    # 명시적 메모리 해제
    del out_ctx, out_ego, maps_context, maps_ego_only, gt_map
    torch.cuda.empty_cache()
    gc.collect()

# 최종 저장 및 평균 출력
df_fin.to_pickle(SAVE_FILENAME)
print("\n" + "="*50 + "\n🎉 모든 실험 완료!\n" + "="*50)