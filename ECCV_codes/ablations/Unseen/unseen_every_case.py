#  nohup python -u unseen_every_case.py >> 3B_unseen_every_case.log 2>&1 & 
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

# ------------------------------------------------------
# 1. 환경 설정 및 모델 로딩
# ------------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ENABLE_SDPA"] = "1"

TARGET_ROOT = f"{AGD20K_PATH}/Unseen/testset/egocentric"
GT_ROOT = f"{AGD20K_PATH}/Unseen/testset/GT"
SAVE_FILENAME = "attention_result_unseen_5exp.pkl"
VIS_DIR = "result_vis_5exp"
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
trackers = {
    'Exp1': MetricsTracker(name="Exp1_LastInput"),
    'Exp2': MetricsTracker(name="Exp2_AvgOutput"),
    'Exp3': MetricsTracker(name="Exp3_Top1_Raw"),
    'Exp4': MetricsTracker(name="Exp4_Top1_Obj"),
    'Exp5': MetricsTracker(name="Exp5_Top1_Adapt"),
    'Exp6': MetricsTracker(name="Exp6_PLSP")
}

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

df_fin = pd.DataFrame(data_list)

# 실험별 메트릭 컬럼 초기화
exp_names = ['Exp1_LastInput', 'Exp2_AvgOutput', 'Exp3_Top1_Raw', 'Exp4_Top1_Obj', 'Exp5_Top1_Adapt']
metrics_keys = ['KLD', 'SIM', 'NSS']
for exp in exp_names:
    for metric in metrics_keys:
        df_fin[f"{exp}_{metric}"] = None
df_fin['top_token_text'] = None 

print(f"✅ 총 {len(df_fin)}개의 데이터 준비 완료.")

# ------------------------------------------------------
# 5. Main Loop
# ------------------------------------------------------
system_prompt = "You are a helpful language and vision assistant."

for index, row in tqdm(df_fin.iterrows(), total=len(df_fin), desc="Processing"):
    object_name = row['object'].replace("_", " ")
    action = row['action']
    file_path = row['full_path']
    gt_path = row['gt_path']
    filename = row['filename']
    PLSP_name = prompt_dict_obj[action][row['object']]
    
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
    
    # Attention Extraction
    with torch.no_grad():
        outputs = model(input_ids=generated_ids, pixel_values=inputs.pixel_values, image_grid_thw=inputs.image_grid_thw, attention_mask=torch.ones_like(generated_ids), output_attentions=True, return_dict=True)
    attentions = outputs.attentions
    
    ids_list = generated_ids[0].tolist()
    tok = processor.tokenizer
    
    if tok.convert_tokens_to_ids("<|vision_start|>") not in ids_list: continue
    vis_start_idx = ids_list.index(tok.convert_tokens_to_ids("<|vision_start|>"))
    vis_end_idx   = ids_list.index(tok.convert_tokens_to_ids("<|vision_end|>"))
    
    grid_t, grid_h, grid_w = inputs.image_grid_thw[0].detach().cpu().numpy()
    llm_grid_h, llm_grid_w = grid_h // 2, grid_w // 2 

    # -------------------------------------------------------------------------
    # [PRE-CALCULATION] Prepare Base Maps
    # -------------------------------------------------------------------------
    clip_object_heatmap, clip_object_mask = get_clipseg_mask(file_path, object_name, llm_grid_h, llm_grid_w)
    clip_plsp_heatmap, clip_plsp_mask = get_clipseg_mask(file_path, PLSP_name, llm_grid_h, llm_grid_w)

    # 1. Last Token of Input
    last_input_idx = input_len - 1
    map_last_input = torch.zeros((llm_grid_h * llm_grid_w), device=device)
    for layer_attn in attentions:
        heads_attn = layer_attn[0, :, last_input_idx, vis_start_idx+1 : vis_end_idx]
        map_last_input += heads_attn.sum(dim=0)
    np_map_last_input = map_last_input.reshape(llm_grid_h, llm_grid_w).cpu().numpy().astype(np.float32)

    # 2. Avg Output & Top-1
    map_avg_accum = torch.zeros((llm_grid_h * llm_grid_w), device=device)
    token_candidates = []
    token_count = 0
    
    for q_idx in range(input_len, len(ids_list)):
        token_heatmap_step = torch.zeros((llm_grid_h * llm_grid_w), device=device)
        for layer_attn in attentions:
            heads_attn = layer_attn[0, :, q_idx, vis_start_idx+1 : vis_end_idx]
            token_heatmap_step += heads_attn.sum(dim=0)
        
        map_avg_accum += token_heatmap_step
        token_count += 1
        
        heatmap_np = token_heatmap_step.reshape(llm_grid_h, llm_grid_w).cpu().numpy().astype(np.float32)
        score = (heatmap_np * clip_object_mask).sum() 
        
        token_candidates.append({
            "idx": q_idx,
            "str": tok.decode([ids_list[q_idx]]),
            "score": score,
            "heatmap": heatmap_np
        })
        
    if token_count > 0:
        map_avg_output = (map_avg_accum / token_count).reshape(llm_grid_h, llm_grid_w).cpu().numpy().astype(np.float32)
    else:
        map_avg_output = np.zeros((llm_grid_h, llm_grid_w), dtype=np.float32)

    if token_candidates:
        top_token = sorted(token_candidates, key=lambda x: x['score'], reverse=True)[0]
        map_top1 = top_token['heatmap']
        top_token_text = top_token['str']
        
        next_idx = top_token['idx'] + 1
        following_text = ""
        for cand in token_candidates:
            if cand['idx'] == next_idx:
                following_text = cand['str']
                break
    else:
        map_top1 = np.zeros((llm_grid_h, llm_grid_w), dtype=np.float32)
        top_token_text = "None"
        following_text = ""

    # -------------------------------------------------------------------------
    # [EXPERIMENTS] Generate Final Maps
    # -------------------------------------------------------------------------
    final_maps = {} 
    final_maps['Exp1'] = apply_post_processing(np_map_last_input.copy(), refinement_heatmap=None, w=w, h=h)
    final_maps['Exp2'] = apply_post_processing(map_avg_output.copy(), refinement_heatmap=None, w=w, h=h)
    final_maps['Exp3'] = apply_post_processing(map_top1.copy(), refinement_heatmap=None, w=w, h=h)
    final_maps['Exp4'] = apply_post_processing(map_top1.copy(), refinement_heatmap=clip_object_heatmap, w=w, h=h)
    
    refined_prompt = f"{top_token_text} {following_text}".replace('.', '').strip()
    clip_specific_heatmap, clip_specific_mask = get_clipseg_mask(file_path, refined_prompt, llm_grid_h, llm_grid_w)
    
    if check_heatmap_containment(clip_specific_mask, clip_object_mask):
        adaptive_heatmap = clip_specific_heatmap 
        exp5_label = f"Part('{refined_prompt}')"
    else:
        adaptive_heatmap = clip_object_heatmap
        exp5_label = f"Obj('{object_name}')"
    final_maps['Exp5'] = apply_post_processing(map_top1.copy(), refinement_heatmap=adaptive_heatmap, w=w, h=h)
    final_maps['Exp6'] = apply_post_processing(map_top1.copy(), refinement_heatmap=clip_plsp_heatmap, w=w, h=h)

    # -------------------------------------------------------------------------
    # [EVALUATION] Calculate Metrics & UPDATE TRACKERS
    # -------------------------------------------------------------------------
    gt_map = load_ground_truth(gt_path) 
    
    if gt_map is not None:
        if gt_map.shape != (h, w):
                gt_map = cv2.resize(gt_map, (w, h), interpolation=cv2.INTER_NEAREST)

        # 실험 키 매핑 (내부 키 -> 데이터프레임 컬럼 프리픽스)
        key_map = {
            'Exp1': 'Exp1_LastInput',
            'Exp2': 'Exp2_AvgOutput',
            'Exp3': 'Exp3_Top1_Raw',
            'Exp4': 'Exp4_Top1_Obj',
            'Exp5': 'Exp5_Top1_Adapt',
            'Exp6': 'Exp6_PLSP'
        }

        print(f"\n--- Metrics Update [{index}] ---")
        for exp_key, pred_map in final_maps.items():
            col_prefix = key_map[exp_key]
            
            # 메트릭 계산
            metrics = calculate_metrics(pred_map, gt_map)
            
            # 1. 데이터프레임 저장
            df_fin.at[index, f"{col_prefix}_KLD"] = metrics['KLD']
            df_fin.at[index, f"{col_prefix}_SIM"] = metrics['SIM']
            df_fin.at[index, f"{col_prefix}_NSS"] = metrics['NSS']

            # 2. ★ Tracker 업데이트 및 출력
            tracker = trackers[exp_key]
            tracker.update(metrics)
            tracker.print_metrics(metrics, filename)

    df_fin.at[index, 'top_token_text'] = top_token_text

    # -------------------------------------------------------------------------
    # [VISUALIZATION] Save Comparison Plot
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 8, figsize=(28, 4))
    axes[0].imshow(orig_img); axes[0].set_title(f"Original\n{object_name}"); axes[0].axis('off')
    
    if gt_map is not None:
        axes[1].imshow(gt_map, cmap='gray'); axes[1].set_title("GT")
    else:
        axes[1].set_title("No GT")
    axes[1].axis('off')
    
    exp_titles = ["Exp1 LastIn", "Exp2 AvgOut", "Exp3 Top1", "Exp4 Top1+Obj", f"Exp5 {exp5_label}", "Exp6 PLSP"]
    for i, (key, title) in enumerate(zip(['Exp1', 'Exp2', 'Exp3', 'Exp4', 'Exp5', 'Exp6'], exp_titles)):
        ax = axes[i+2]
        ax.imshow(final_maps[key], cmap='jet', vmin=0, vmax=1)
        ax.set_title(title, fontsize=10); ax.axis('off')
        
        # 메트릭 표시
        if gt_map is not None:
            tracker = trackers[key] # 현재 트래커의 평균값 가져오기 가능
            sim = df_fin.at[index, f"{key_map[key]}_SIM"]
            nss = df_fin.at[index, f"{key_map[key]}_NSS"]
            ax.text(0.5, -0.1, f"S:{sim:.2f} N:{nss:.2f}", transform=ax.transAxes, ha='center', fontsize=9, color='blue')

    save_path = os.path.join(VIS_DIR, f"{action}_{object_name}_{filename.split('.')[0]}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Memory Cleanup
    # -------------------------------------------------------------------------
    if index % 50 == 0:
        df_fin.to_pickle(SAVE_FILENAME)
    
    del generated_ids, outputs, attentions, token_candidates, final_maps
    torch.cuda.empty_cache()
    gc.collect()



df_fin.to_pickle(SAVE_FILENAME)
print(f"🎉 모든 실험 완료! 결과 저장: {SAVE_FILENAME}")

# 최종 평균 출력
print("\n📊 Final Average Metrics:")
for key, tracker in trackers.items():
    print(f"[{key}] KLD: {tracker.KLD.avg:.4f} | SIM: {tracker.SIM.avg:.4f} | NSS: {tracker.NSS.avg:.4f}")