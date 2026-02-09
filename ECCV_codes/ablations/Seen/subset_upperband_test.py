# nohup python -u subset_upperband_test.py >> 2b_subset_upper_bound_analysis.log 2>&1 &

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
from config import AGD20K_PATH, model_name

# ------------------------------------------------------
# 1. 환경 설정
# ------------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ENABLE_SDPA"] = "1"

TARGET_JSON_PATH = "/home/bongo/porter_notebook/research/qwen3/selected_samples.json"
EXO_CACHE_PATH = "fixed_exo_candidates.pkl"  # [변경] 미리 생성한 캐시 파일 경로
EXO_ROOT_BASE = Path(f"{AGD20K_PATH}/Seen/trainset/exocentric")

SAVE_PKL_NAME = "2B_upper_bound_kld_best.pkl"
SAVE_ALL_PKL_NAME = "2B_all_trials_metrics.pkl"
VIS_ROOT = "2B_vis_upper_bound_analysis"
VIS_TRIALS_DIR = os.path.join(VIS_ROOT, "1_all_trials")
VIS_BEST_DIR = os.path.join(VIS_ROOT, "2_best_comparison")
VIS_WORST_DIR = os.path.join(VIS_ROOT, "3_worst_comparison")

os.makedirs(VIS_TRIALS_DIR, exist_ok=True)
os.makedirs(VIS_BEST_DIR, exist_ok=True)
os.makedirs(VIS_WORST_DIR, exist_ok=True)

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
with open(TARGET_JSON_PATH, 'r') as f:
    json_data = json.load(f)

# [변경] Exo Cache 로드
if not os.path.exists(EXO_CACHE_PATH):
    print(f"❌ Error: Cache file {EXO_CACHE_PATH} not found. Run make_exo_cache.py first.")
    exit() # try-except 쓰지 말래서 그냥 exit

with open(EXO_CACHE_PATH, 'rb') as f:
    exo_cache_data = pickle.load(f)
print(f"✅ Loaded Exo Cache for {len(exo_cache_data)} cases.")

target_samples = []
for key, item in json_data["selected_samples"].items():
    img_path = item["image_path"].replace("${AGD20K_PATH}", AGD20K_PATH)
    gt_path = img_path.replace("egocentric", "GT").replace(".jpg", ".png")
    target_samples.append({
        "case_id": key,
        "action": item["action"],
        "object": item["object"],
        "ego_path": img_path,
        "gt_path": gt_path,
        "filename": os.path.basename(img_path)
    })

df_results = pd.DataFrame(target_samples)

# ------------------------------------------------------
# 4. Main Loop & Accumulators
# ------------------------------------------------------
system_prompt = "You are a helpful language and vision assistant."
v_start_token = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
v_end_token = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

# [NEW] 누적 메트릭 저장을 위한 변수 초기화
cum_metrics = {
    'ego': {'KLD': 0.0, 'SIM': 0.0, 'NSS': 0.0},
    'best': {'KLD': 0.0, 'SIM': 0.0, 'NSS': 0.0},
    'worst': {'KLD': 0.0, 'SIM': 0.0, 'NSS': 0.0}
}
valid_count = 0

all_trials_rows = []

for index, row in tqdm(df_results.iterrows(), total=len(df_results), desc="Upper Bound Analysis"):
    case_id = row['case_id']
    action = row['action']
    object_name = row['object']
    ego_path = row['ego_path']
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
        


    # --- [STEP 1] Exo 이미지 후보군 추론 (Upper Bound Search) ---
    exo_dir = EXO_ROOT_BASE / action / object_name
    valid_ext = {'.jpg', '.jpeg', '.png'}
    
    if not exo_dir.exists(): continue
    all_exo_files = [p for p in exo_dir.rglob("*") if p.suffix.lower() in valid_ext]
    if not all_exo_files: continue

    exo_candidates = exo_cache_data[case_id] # 파일 경로 리스트 (str)
    
    best_kld_score = 999.0
    worst_kld_score = 0.0
    best_exo_data = None
    
    case_trial_dir = os.path.join(VIS_TRIALS_DIR, f"{action}_{object_name}_{row['filename'].split('.')[0]}")
    os.makedirs(case_trial_dir, exist_ok=True)
    
    desc_context = f"Refer to the second image (exocentric view) for context. Based on the first image (egocentric view), when people perform {action} with {object_name.replace('_',' ')}, which part of the {object_name.replace('_',' ')} is used for '{action}'? Answer in one sentence."

    for idx, exo_file in enumerate(exo_candidates):
        exo_file = exo_file.replace("/home/DATA/AGD20K", AGD20K_PATH )
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

        # [NEW] 전체 트라이얼 데이터 수집
        all_trials_rows.append({
            "case_id": case_id,
            "action": action,
            "object": object_name,
            "ego_image": row['filename'],
            "exo_image": os.path.basename(exo_file),
            "ego_path": ego_path,
            "exo_path": exo_file,
            "kld": metrics_exo['KLD'],
            "sim": metrics_exo['SIM'],
            "nss": metrics_exo['NSS']
        })
        filename = os.path.basename(exo_file)
        
        # Save Trial Vis
        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        ax[0].imshow(orig_img); ax[0].axis('off')
        ax[1].imshow(Image.open(exo_file).convert("RGB")); ax[1].axis('off')
        ax[2].imshow(map_exo, cmap='jet'); ax[2].set_title(f"KLD: {metrics_exo['KLD']:.3f}"); ax[2].axis('off')
        ax[3].imshow(gt_map, cmap='gray'); ax[3].axis('off')
        plt.savefig(os.path.join(case_trial_dir, f"trial_{idx}_{os.path.basename(exo_file)}.png"))
        plt.close(fig)
        
        if metrics_exo['KLD'] < best_kld_score:
            best_kld_score = metrics_exo['KLD']
            best_exo_data = {
                'map': map_exo, 
                'metrics': metrics_exo, 
                'filename': os.path.basename(exo_file),
                'path': str(exo_file)  # <--- [중요] 나중에 이미지를 열기 위해 경로 저장
            }
        if metrics_exo['KLD'] > worst_kld_score:
            worst_kld_score = metrics_exo['KLD']
            worst_exo_data = {
                'map': map_exo, 
                'metrics': metrics_exo, 
                'filename': os.path.basename(exo_file),
                'path': str(exo_file)  # <--- [중요] 나중에 이미지를 열기 위해 경로 저장
            }
        del in_ctx, out_ctx, gen_ctx
        torch.cuda.empty_cache()
        

    # --- [STEP 2] Best Case 저장 및 누적 메트릭 출력 ---
    if best_exo_data is not None:
        # 1. 누적 업데이트
        valid_count += 1
        for k in ['KLD', 'SIM', 'NSS']:
            cum_metrics['ego'][k] += metrics_ego[k]
            cum_metrics['best'][k] += best_exo_data['metrics'][k]
            cum_metrics['worst'][k] += worst_exo_data['metrics'][k]
            
        # 2. DataFrame 저장
        df_results.at[index, 'ego_kld'] = metrics_ego['KLD']
        df_results.at[index, 'best_exo_kld'] = best_exo_data['metrics']['KLD']
        df_results.at[index, 'worst_exo_kld'] = worst_exo_data['metrics']['KLD']
        
        # 3. [수정 2] 비교 시각화 저장 (Type 2) -> 5개 컬럼으로 확장
        # (순서: Ego원본 -> Best Exo원본 -> GT -> Ego Map -> Best Exo Map)
        best_exo_img_vis = Image.open(best_exo_data['path']).convert("RGB") # 경로에서 이미지 로드
        
        fig, axes = plt.subplots(1, 5, figsize=(25, 6)) # 1x5 그리드로 변경
        
        # [0] Ego Image
        axes[0].imshow(orig_img)
        axes[0].set_title(f"Ego: {action}-{object_name}")
        axes[0].axis('off')
        
        # [1] Best Exo Image (New!)
        axes[1].imshow(best_exo_img_vis)
        axes[1].set_title(f"Best Exo Ref\n{best_exo_data['filename']}")
        axes[1].axis('off')

        # [2] GT Map
        axes[2].imshow(gt_map, cmap='gray')
        axes[2].set_title("GT")
        axes[2].axis('off')
        
        # [3] Ego Heatmap
        axes[3].imshow(map_ego, cmap='jet')
        axes[3].set_title(f"Ego Only\nK:{metrics_ego['KLD']:.3f} S:{metrics_ego['SIM']:.2f}", color='red')
        axes[3].axis('off')
        
        # [4] Best Exo Heatmap
        axes[4].imshow(best_exo_data['map'], cmap='jet')
        axes[4].set_title(f"Best Exo Map\nK:{best_exo_data['metrics']['KLD']:.3f} S:{best_exo_data['metrics']['SIM']:.2f}", color='blue')
        axes[4].axis('off')
        
        save_name = f"COMP_{action}_{object_name}_{row['filename'].split('.')[0]}.png"
        plt.savefig(os.path.join(VIS_BEST_DIR, save_name))
        plt.close(fig)
        
        worst_exo_img_vis = Image.open(worst_exo_data['path']).convert("RGB") # 경로에서 이미지 로드
        
        fig, axes = plt.subplots(1, 5, figsize=(25, 6)) # 1x5 그리드로 변경
        
        # [0] Ego Image
        axes[0].imshow(orig_img)
        axes[0].set_title(f"Ego: {action}-{object_name}")
        axes[0].axis('off')
        
        # [1] Best Exo Image (New!)
        axes[1].imshow(worst_exo_img_vis)
        axes[1].set_title(f"Worst Exo Ref\n{worst_exo_data['filename']}")
        axes[1].axis('off')

        # [2] GT Map
        axes[2].imshow(gt_map, cmap='gray')
        axes[2].set_title("GT")
        axes[2].axis('off')
        
        # [3] Ego Heatmap
        axes[3].imshow(map_ego, cmap='jet')
        axes[3].set_title(f"Ego Only\nK:{metrics_ego['KLD']:.3f} S:{metrics_ego['SIM']:.2f}", color='red')
        axes[3].axis('off')
        
        # [4] Worst Exo Heatmap
        axes[4].imshow(worst_exo_data['map'], cmap='jet')
        axes[4].set_title(f"Worst Exo Map\nK:{worst_exo_data['metrics']['KLD']:.3f} S:{worst_exo_data['metrics']['SIM']:.2f}", color='blue')
        axes[4].axis('off')
        
        save_name = f"COMP_{action}_{object_name}_{row['filename'].split('.')[0]}.png"
        plt.savefig(os.path.join(VIS_WORST_DIR, save_name))
        plt.close(fig)


        # 4. 실시간 누적 평균 출력 (동일)
        avg_ego = {k: v/valid_count for k, v in cum_metrics['ego'].items()}
        avg_best = {k: v/valid_count for k, v in cum_metrics['best'].items()}
        avg_worst = {k: v/valid_count for k, v in cum_metrics['worst'].items()}
        
        print(f"\n📊 [Avg Metrics @ {valid_count}]")
        print(f"   Baseline (Ego)   : KLD {avg_ego['KLD']:.3f} | SIM {avg_ego['SIM']:.3f} | NSS {avg_ego['NSS']:.3f}")
        print(f"   UpperBnd (Best) : KLD {avg_best['KLD']:.3f} | SIM {avg_best['SIM']:.3f} | NSS {avg_best['NSS']:.3f}")
        print(f"   LowerBnd (Worst): KLD {avg_worst['KLD']:.3f} | SIM {avg_worst['SIM']:.3f} | NSS {avg_worst['NSS']:.3f}")
        print(f"   🚀 Gain (Ego-Best): KLD {avg_ego['KLD'] - avg_best['KLD']:.3f} (Lower is better)")
        print("-" * 60)

    if index % 5 == 0:
        df_results.to_pickle(SAVE_PKL_NAME)
        pd.DataFrame(all_trials_rows).to_pickle(SAVE_ALL_PKL_NAME)

df_results.to_pickle(SAVE_PKL_NAME)
pd.DataFrame(all_trials_rows).to_pickle(SAVE_ALL_PKL_NAME) # [NEW] 전체 저장
print("\n🎉 Analysis Complete!")