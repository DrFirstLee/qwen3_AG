
import json
import os
import ast
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.cm as cm  # 컬러맵 사용을 위해 필요

from torchvision import transforms
from PIL import Image
from sklearn.cluster import KMeans
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# CLIPSeg 모델 로드
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
clip_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

# ==========================================
# 1. 경로 설정 (환경에 맞게 수정해주세요)
# ==========================================
JSON_PATH = '/home/bongo/porter_notebook/research/qwen3/result_32B_relative/results_vlm_bbox_relative.json'

# [중요] 원본 이미지가 들어있는 폴더 경로를 입력하세요.
# 예: '/home/bongo/dataset/images/'
SOURCE_IMAGE_DIR = "/home/DATA/AGD20K/Seen/testset/egocentric"

OUTPUT_DIR = 'bbox_clipseg_32B_relative'
epsilon = 0.1#1e-6 #
# ==========================================
# 2. BBox 파싱 헬퍼 함수
# ==========================================
def parse_bbox_str(bbox_str):
    """
    다양한 포맷의 문자열에서 [x1, y1, x2, y2] 리스트를 추출합니다.
    Case 1: "[275, 227, 899, 688]"
    Case 2: "```json\n[\n\t{\"bbox_2d\": [438, ...], ...}\n]\n```"
    """
    # 1. Markdown 코드 블록 기호 제거 (```json, ```)
    clean_str = bbox_str.replace("```json", "").replace("```", "").strip()
    
    # 2. JSON 파싱 시도 (객체 형태인 경우)
    try:
        # JSON으로 로드
        data = json.loads(clean_str)
        
        # 리스트 안에 딕셔너리가 있는 구조인지 확인 [{"bbox_2d": ...}]
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            if "bbox_2d" in data[0]:
                return data[0]["bbox_2d"]
    except (json.JSONDecodeError, KeyError, TypeError):
        pass # JSON 실패 시 다음 단계(단순 리스트)로 넘어감

    # 3. 단순 리스트 파싱 시도 ("[1, 2, 3, 4]")
    try:
        return ast.literal_eval(clean_str)
    except (ValueError, SyntaxError):
        return None

# 저장할 폴더가 없으면 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# JSON 로드
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"총 {len(data)}개의 데이터를 처리합니다...")

gamma = 0.1
def get_clipseg_heatmap(
        image_path: str,
        model, 
        processor, 
        object_name: str,
    ):
    """
    (수정됨) CLIPSeg 모델을 사용하여 이미지와 텍스트 프롬프트 간의
    세그멘테이션 히트맵을 추출합니다.
    """
    if model is None or processor is None:
        print("Error: CLIPSeg model or processor not loaded.")
        return None, None
    
    original_image = Image.open(image_path).convert('RGB')
    original_size = original_image.size # (width, height)

    # 1. 단일 텍스트 프롬프트 정의
    prompt_text = object_name

    # 2. 입력 처리
    inputs = processor(
        text=[prompt_text], 
        images=[original_image], 
        padding="max_length", 
        return_tensors="pt"
    )
    
    # 3. 예측
    with torch.no_grad():
        outputs = model(**inputs)
        # preds의 shape 처리는 로직에 따라 다르지만, 결과적으로 heatmap을 뽑을 때 주의해야 합니다.
        preds = outputs.logits.unsqueeze(0).unsqueeze(1) 

    # 4. 히트맵 생성
    # [중요 수정] .squeeze()를 추가하여 (1, 352, 352) -> (352, 352)로 변환합니다.
    heatmap_small = torch.sigmoid(preds[0][0]).cpu().detach().squeeze() 
    
    # 디버깅용 출력 (이제 torch.Size([352, 352])가 나와야 함)
    # print(f"shape of clip_heatmap : {heatmap_small.shape}")

    # 5. PIL 이미지 변환 및 리사이즈
    # heatmap_small.numpy()는 이제 (352, 352)이므로 PIL이 정상적으로 인식합니다.
    # float32 타입 유지를 위해 mode='F'를 명시할 수도 있으나, 보통 그냥 넘겨도 됩니다.
    final_heatmap = np.array(
        Image.fromarray(heatmap_small.numpy())
        .resize(original_size, resample=Image.Resampling.BILINEAR)
    )
    
    # print(f"shape of final_heatmap : {final_heatmap.shape}")

    # 0-1 정규화
    if final_heatmap.max() > 0:
        final_heatmap = (final_heatmap - final_heatmap.min()) / (final_heatmap.max() - final_heatmap.min())
        # gamma, epsilon은 외부 변수를 사용하므로 함수 인자로 받거나 전역 변수여야 합니다.
        # 여기서는 코드 맥락상 전역 변수 gamma, epsilon을 사용한다고 가정합니다.
        final_heatmap = final_heatmap #** gamma + epsilon
        
    return final_heatmap


def get_gt_path(image_path):
    """
    Convert image path to corresponding GT path
    Example:
    /path/Seen/testset/egocentric/action/object/image.jpg
    -> /path/Seen/testset/GT/action/object/image.png
    """
    parts = image_path.split('/')
    # Find the index of 'testset' in the path
    testset_idx = parts.index('testset')
    # Replace 'egocentric' with 'GT' and change extension to .txt
    parts[testset_idx + 1] = 'GT'
    base_name = os.path.splitext(parts[-1])[0]
    parts[-1] = base_name + '.png'
    return '/'.join(parts)

def calculate_metrics(pred_heatmap, gt_map):
    """
    User Provided Metric Function
    """
    # Ensure inputs are proper tensors
    if not isinstance(pred_heatmap, torch.Tensor):
        pred_heatmap = torch.tensor(pred_heatmap)
    if not isinstance(gt_map, torch.Tensor):
        gt_map = torch.tensor(gt_map)
    
    # Flatten tensors and add batch dimension for compatibility
    pred = pred_heatmap.flatten().float().unsqueeze(0)  # [1, H*W]
    gt = gt_map.flatten().float().unsqueeze(0)          # [1, H*W]
    
    eps = 1e-10
    
    # Calculate KLD
    pred_norm = pred / (pred.sum(dim=1, keepdim=True) + eps)
    gt_norm = gt / (gt.sum(dim=1, keepdim=True) + eps)
    # pred_norm += eps # (log에서 에러 방지를 위해 아래처럼 처리 추천)
    kld = F.kl_div((pred_norm + eps).log(), gt_norm, reduction="batchmean").item()
    
    # Calculate SIM
    pred_sim = pred / (pred.sum(dim=1, keepdim=True) + eps)
    gt_sim = gt / (gt.sum(dim=1, keepdim=True) + eps)
    sim = torch.minimum(pred_sim, gt_sim).sum().item() # / len(pred_sim) -> batch 1이므로 생략 가능하나 원본 유지
    
    # Calculate NSS
    pred_nss = pred / (pred.max(dim=1, keepdim=True).values + eps)
    gt_nss = gt / (gt.max(dim=1, keepdim=True).values + eps)
    
    std = pred_nss.std(dim=1, keepdim=True)
    u = pred_nss.mean(dim=1, keepdim=True)
    smap = (pred_nss - u) / (std + eps)
    
    fixation_map = (gt_nss - torch.min(gt_nss, dim=1, keepdim=True).values) / (
        torch.max(gt_nss, dim=1, keepdim=True).values - torch.min(gt_nss, dim=1, keepdim=True).values + eps)
    fixation_map = (fixation_map >= 0.1).float()
    
    nss_values = smap * fixation_map
    nss = nss_values.sum(dim=1) / (fixation_map.sum(dim=1) + eps)
    nss = nss.mean().item()
    
    return {'KLD': kld, 'SIM': sim, 'NSS': nss}

print(f"총 {len(data)}개의 데이터를 평가합니다...")

total_metrics = {'KLD': 0.0, 'SIM': 0.0, 'NSS': 0.0}
valid_count = 0
results_detail = {} # 개별 결과 저장용

for key_str, bbox_raw in data.items():

    # 1. 키 파싱 (object_name$action$file_name)
    # 예: "apple$cut$apple_000054.jpg" -> ["apple", "cut", "apple_000054.jpg"]
    parts = key_str.split('$')
    
    if len(parts) < 3:
        print(f"[Skip] 형식이 맞지 않는 키: {key_str}")
        continue

    object_name = parts[0]
    action = parts[1]
    file_name = parts[2]  # 실제 이미지 파일명



    # --- 2. BBox 값 파싱 (헬퍼 함수 사용) ---
    bbox_1000 = parse_bbox_str(bbox_raw)

    if not bbox_1000 or len(bbox_1000) != 4:
        print(f"[Skip] BBox 파싱 실패 또는 형식 오류: {key_str}")
        continue
    # 3. 원본 이미지 불러오기
    img_full_path = os.path.join(SOURCE_IMAGE_DIR, action, object_name,file_name)
    clip_heatmap = get_clipseg_heatmap(
        img_full_path,
        clip_model, # Pass the model object (now on GPU)
        processor,
        object_name,
    )
    
    gt_path = get_gt_path(img_full_path)  
    
    if not os.path.exists(img_full_path):
        print(f"[Error] 이미지를 찾을 수 없음: {img_full_path}")
        continue

    

    
    img = Image.open(img_full_path).convert("RGB")
    orig_w, orig_h = img.size
    draw = ImageDraw.Draw(img)
    try:
        # GT 열기 (Grayscale L 모드로 변환)    
        with Image.open(gt_path).convert('L') as gt_img:
            # 만약 GT 사이즈가 원본과 다르면 리사이즈
            if gt_img.size != (orig_w, orig_h):
                gt_img = gt_img.resize((orig_w, orig_h), Image.NEAREST)
            
            # Tensor 변환 (0~1 범위로 정규화)
            gt_tensor = torch.from_numpy(np.array(gt_img)).float() / 255.0
    except:
        print("NO GT!!")
        continue
    # --- 4. 좌표 복원 (1000x1000 -> 원본 해상도) ---
    # 공식: model_coord * (orig_len / 1000)
    x1 = int(bbox_1000[0] * (orig_w / 1000))
    y1 = int(bbox_1000[1] * (orig_h / 1000))
    x2 = int(bbox_1000[2] * (orig_w / 1000))
    y2 = int(bbox_1000[3] * (orig_h / 1000))
    
    final_bbox = [x1, y1, x2, y2]
    # 좌표 클리핑 (이미지 범위 안으로 제한)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(orig_w, x2); y2 = min(orig_h, y2)
    # 빈 마스크 생성 (0으로 채움)

    if hasattr(clip_heatmap, 'cpu'):
            heatmap_np = clip_heatmap.detach().cpu().numpy()
    elif isinstance(clip_heatmap, np.ndarray):
        heatmap_np = clip_heatmap
    else:
        # 리스트 등 다른 타입일 경우
        heatmap_np = np.array(clip_heatmap)
    
    heatmap_np = heatmap_np.squeeze()

    # 결과를 담을 빈(0으로 채워진) 텐서 생성
    masked_clip_heatmap = np.zeros_like(heatmap_np)

    # 좌표가 이미지 범위를 벗어나지 않도록 안전장치 (Clamping)
    x1 = max(0, min(x1, orig_w))
    y1 = max(0, min(y1, orig_h))
    x2 = max(0, min(x2, orig_w))
    y2 = max(0, min(y2, orig_h))        
    # 4. BBox 그리기 (x1, y1, x2, y2)
    # outline: 테두리 색상, width: 두께
    if x2 > x1 and y2 > y1:
        # 원본 크기로 늘린 히트맵에서 해당 영역만 가져와서 복사
        masked_clip_heatmap[y1:y2, x1:x2] = clip_heatmap[y1:y2, x1:x2]

    masked_clip_heatmap += epsilon
    # ---------------------------------------------------------
    # [Step 3] 히트맵 시각화 (Colorization & Overlay)
    # ---------------------------------------------------------
    # 1. Normalize (0~1)
    map_np = masked_clip_heatmap
    min_v, max_v = map_np.min(), map_np.max()
    if max_v - min_v > 0:
        norm_map = (map_np - min_v) / (max_v - min_v)
    else:
        norm_map = map_np # 모든 값이 0인 경우

    # 2. Apply Colormap (Jet: Blue -> Red)
    # cm.jet은 RGBA(0~1) 반환 -> 0~255로 변환
    colormap_img = cm.jet(norm_map)
    colormap_img = (colormap_img * 255).astype(np.uint8)
    
    # 3. Create PIL Image
    heatmap_pil = Image.fromarray(colormap_img)

    # 4. Alpha Channel 조절 (중요!)
    # 히트맵 값이 0인 곳(BBox 밖)은 투명하게, 값이 높은 곳은 진하게
    # 최대 투명도를 180(약 70%) 정도로 설정하여 원본이 비치게 함
    alpha = (norm_map * 180).astype(np.uint8)
    heatmap_pil.putalpha(Image.fromarray(alpha))

    # 5. Overlay
    img_rgba = img.convert("RGBA")
    img_rgba.alpha_composite(heatmap_pil) # 합성

    # ---------------------------------------------------------
    # [Step 4] BBox 그리기 및 저장
    # ---------------------------------------------------------
    # 합성이 끝난 이미지 위에 빨간 박스를 그려야 선명하게 보임
    draw = ImageDraw.Draw(img_rgba)
    draw.rectangle(final_bbox, outline="red", width=5)

    # 텍스트 라벨
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
    except:
        font = ImageFont.load_default()
    
    label = f"{action}"
    # 텍스트 배경 (선택사항)
    # draw.rectangle([x1, y1-35, x1+100, y1], fill="red")
    draw.text((x1, y1-35), label, fill="red", font=font)

    # 저장
    save_name = f"{object_name}_{action}_{file_name}"
    save_path = os.path.join(OUTPUT_DIR, save_name)
    
    # RGBA -> RGB 변환 후 저장 (JPEG 저장을 위해)
    img_rgba.convert("RGB").save(save_path)
    print(f"[Saved] {save_path}")
    # 6. Metric 계산
    metrics = calculate_metrics(masked_clip_heatmap, gt_tensor)
    
    # 누적
    total_metrics['KLD'] += metrics['KLD']
    total_metrics['SIM'] += metrics['SIM']
    total_metrics['NSS'] += metrics['NSS']
    valid_count += 1
    # (선택) 개별 결과 기록
    results_detail[key_str] = metrics

    # ==========================================
    # 4. 결과 출력
    # ==========================================
    print("-" * 50)
    if valid_count > 0:
        avg_kld = total_metrics['KLD'] / valid_count
        avg_sim = total_metrics['SIM'] / valid_count
        avg_nss = total_metrics['NSS'] / valid_count
        
        print(f"Evaluation Results (Total: {valid_count})")
        print(f"KLD (Lower is better): {avg_kld:.4f}")
        print(f"SIM (Higher is better): {avg_sim:.4f}")
        print(f"NSS (Higher is better): {avg_nss:.4f}")
        
        # 결과 파일 저장
        summary = {
            "average": {"KLD": avg_kld, "SIM": avg_sim, "NSS": avg_nss},
            "details": results_detail
        }
        with open("bbox_clipseg_metrics_result.json", "w") as f:
            json.dump(summary, f, indent=4)