import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import textwrap
import os
import numpy  as np
import pandas as pd
import cv2
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import timm
import glob
import torch
from collections import Counter
import textwrap

import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import transforms

AGD20K_PATH = '/home/DATA/AGD20K'
# 현재 파일 위치 기준 상위 폴더를 시스템 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# ------------------------------------------------------
# 3. Local Modules (사용자 정의 모듈)
# ------------------------------------------------------
# 경로 설정이 완료된 후 import 해야 합니다.
from VLM_model_dot_relative import MetricsTracker
from file_managing import (
    load_selected_samples,
    get_actual_path,
    get_gt_path,
    prompt_dict_obj
)



from scipy.stats import pearsonr
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
clip_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
metrics_tracker_dino = MetricsTracker(name="only_ego")


def load_ground_truth( gt_path):
    """
    Load and process ground truth image
    Args:
        gt_path (str): Path to the ground truth image
    Returns:
        torch.Tensor: Processed ground truth tensor normalized to [0, 1]
    """

    # Load the ground truth image
    gt_img = Image.open(gt_path)

    # Convert to grayscale if image is RGB
    if gt_img.mode == 'RGB':
        gt_img = gt_img.convert('L')

    # Convert to tensor
    gt_tensor = transforms.ToTensor()(gt_img).squeeze(0)

    # Normalize to [0, 1]
    if gt_tensor.max() > 0:
        gt_tensor = (gt_tensor - gt_tensor.min()) / (gt_tensor.max() - gt_tensor.min())

    return gt_tensor
    

def calculate_metrics( pred_heatmap, gt_map):
    """
    Calculate comparison metrics between predicted heatmap and GT (following original metric.py)
    Args:
        pred_heatmap (torch.Tensor): Predicted heatmap
        gt_map (torch.Tensor): Ground truth map
    Returns:
        dict: Dictionary containing KLD, SIM, and NSS metrics
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

    # Calculate KLD following original implementation
    # Normalize to probability distributions
    pred_norm = pred / pred.sum(dim=1, keepdim=True)
    gt_norm = gt / gt.sum(dim=1, keepdim=True)
    pred_norm += eps
    kld = F.kl_div(pred_norm.log(), gt_norm, reduction="batchmean").item()

    # Calculate SIM following original implementation
    pred_sim = pred / pred.sum(dim=1, keepdim=True)
    gt_sim = gt / gt.sum(dim=1, keepdim=True)
    sim = torch.minimum(pred_sim, gt_sim).sum().item() / len(pred_sim)

    # Calculate NSS following original implementation
    # First normalize by max values
    pred_nss = pred / pred.max(dim=1, keepdim=True).values
    gt_nss = gt / gt.max(dim=1, keepdim=True).values

    # Calculate z-score for prediction
    std = pred_nss.std(dim=1, keepdim=True)
    u = pred_nss.mean(dim=1, keepdim=True)
    smap = (pred_nss - u) / (std + eps)

    # Create fixation map from GT
    fixation_map = (gt_nss - torch.min(gt_nss, dim=1, keepdim=True).values) / (
        torch.max(gt_nss, dim=1, keepdim=True).values - torch.min(gt_nss, dim=1, keepdim=True).values + eps)
    fixation_map = (fixation_map >= 0.1).float()

    # Calculate NSS
    nss_values = smap * fixation_map
    nss = nss_values.sum(dim=1) / (fixation_map.sum(dim=1) + eps)
    nss = nss.mean().item()

    return {
        'KLD': kld,
        'SIM': sim,
        'NSS': nss
    }

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
        final_heatmap = final_heatmap ## ** gamma ##+ epsilon
        
    return final_heatmap

# Min-Max 표준화 함수 정의
def min_max_normalize(arr):
    denom = arr.max() - arr.min()
    if denom == 0:
        return np.zeros_like(arr)
    return (arr - arr.min()) / (denom + 1e-8)

# 시각화할 타겟 헤드 리스트 정의 - correl
# target_heads = [(26, 20), (26, 33), (24, 31)]

# target_heads = [(26, 20), (22, 26), (0, 20)]# top3 correl

target_heads = [(15, 1), (14, 6), (28, 42)] # top3-correl / simple description


# Top 1 | Layer:  0, Head: 20 | 선정 횟수: 104239회 (전체의 21.0%)
# Top 2 | Layer: 15, Head: 12 | 선정 횟수: 93906회 (전체의 18.9%)
# Top 3 | Layer: 15, Head: 34 | 선정 횟수: 92745회 (전체의 18.7%)
# Top 4 | Layer:  0, Head: 17 | 선정 횟수: 87156회 (전체의 17.6%)
# Top 5 | Layer: 16, Head: 59 | 선정 횟수: 73930회 (전체의 14.9%)
# target_heads = [(0, 20), (15, 12), (15, 34)] # bin

# target_heads = [(0, 20), (0, 17), (12, 25)] # top3-bin



df_attention = pd.read_pickle("attention_result_32B_simple_description.pkl")

for index, row in df_attention.iterrows():
    object_name = row['object']
    action = row['action']
    filename = row['filename']
    # description = row['description']
    description = f"when people perform {action} with {object_name}, which part of the {object_name} is used for '{action}'?"

    attention_value = row['s_img']
    file_name_real = f"{AGD20K_PATH}/Seen/testset/egocentric/{action}/{object_name}/{filename}"
    gt_path =  f"{AGD20K_PATH}/Seen/testset/GT/{action}/{object_name}/{filename.split('.')[0]}.png"
    print(f"Processing image {index}: Combining 3 Heads...")

    clip_heatmap = get_clipseg_heatmap(
        file_name_real,
        clip_model, # Pass the model object (now on GPU)
        processor,
        object_name,
    )
    # 2. CLIPSeg 히트맵을 0,1의 바이너리마스크로 만듬
    clip_binary_mask = (clip_heatmap > 0.15).astype(np.float32)

    # 1. 원본 이미지 로드
    orig_img = cv2.imread(file_name_real)
    if orig_img is None: continue
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    h, w, _ = orig_img.shape

    # 2. 지정된 3개 헤드의 히트맵 추출 및 합산
    combined_heatmap_raw = np.zeros((31, 31), dtype=np.float32)
    found_count = 0

    for idx in attention_value:
        curr_key = (idx['layer'], idx['head'])
        if curr_key in target_heads:
            # 각 헤드를 개별적으로 Min-Max 정규화 후 더함
            norm_map = min_max_normalize(idx['heatmap'])
            combined_heatmap_raw += norm_map
            found_count += 1

    # 3개 헤드를 모두 찾은 경우에만 진행
    if found_count < 3:
        print(f"⚠️ Index {index}: 타겟 헤드를 모두 찾지 못했습니다. (찾은 개수: {found_count})")
        continue

    # 3. 합산된 히트맵을 다시 0~1 사이로 정규화 (최종 시각화용)
    final_combined_norm = min_max_normalize(combined_heatmap_raw)

    # ✨ [추가] 상위 30% 값만 남기기 (70th Percentile)
    # 70퍼센타일 값을 찾습니다 (이 값보다 큰 값이 상위 30%에 해당)
    threshold = np.percentile(final_combined_norm, 97)

    # 임계값보다 작은 값은 모두 0으로 만듭니다.
    final_combined_norm[final_combined_norm < threshold] = 0

    # (선택사항) 0이 된 배경을 제외하고 나머지 영역을 다시 0~1로 정규화하면 더 선명해집니다.
    # final_combined_norm = min_max_normalize(final_combined_norm)

    
    
    final_combined_norm = min_max_normalize(final_combined_norm)
    # final_combined_norm += 0.001
    
    # 4. Final image resize from path to real image size
    attn_resized = cv2.resize(final_combined_norm, (w, h), interpolation=cv2.INTER_LINEAR)

    ## Gausian Blur!!!
    k_val = int(min(w, h) * 0.5) 
    if k_val % 2 == 0: 
        k_val += 1
    kernel_size = (k_val, k_val)
    attn_resized = cv2.GaussianBlur(attn_resized, kernel_size, 0)
    attn_resized = min_max_normalize(attn_resized)

    ## CLIP MASK
    # attn_resized = attn_resized * clip_binary_mask

    attn_resized += 0.001
    heatmap_color = cv2.applyColorMap(np.uint8(255 * attn_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # 5. 합성 이미지 생성
    overlay_img = cv2.addWeighted(orig_img, 0.6, heatmap_color, 0.4, 0)
    # print(gt_path)
    gt_map = load_ground_truth(gt_path)
    if gt_map is not None:
        metrics_dino  = calculate_metrics(attn_resized, gt_map)
        metrics_tracker_dino.update(metrics_dino)
    else:
        print("NO GT!!!")
        continue
    metrics_tracker_dino.print_metrics(metrics_dino, filename)
    
    metrics_text = f"[{object_name} {action} {filename}]  KLD: {metrics_dino['KLD']:.4f} | SIM: {metrics_dino['SIM']:.4f} | NSS: {metrics_dino['NSS']:.4f}"
    


    # 6. 시각화 레이아웃 설정
# 6. 시각화 레이아웃 설정 (1x4 구조로 변경 및 figsize 확대)
    # 가로 폭을 22에서 28로 늘려 4개의 그림이 찌그러지지 않게 합니다.
    fig, axes = plt.subplots(1, 5, figsize=(28, 10))
    wrapped_desc = "\n".join(textwrap.wrap(f"Description: {description}", width=130)) # 폭이 넓어져서 width도 약간 늘림
    full_title = (
        f"Object: {object_name}  |  Action: {action}  |  File: {filename}\n"
        f"Combined Attention: (L0 H20) + (L15 H12) + (L15 H34) with CLIP Masking\n"
        f"{wrapped_desc}\n"
        f"{metrics_text}"
    )

    plt.suptitle(full_title, fontsize=16, fontweight='bold', y=0.97)

    # --- 첫 번째: 원본 이미지 ---
    axes[0].imshow(orig_img)
    axes[0].set_title("1. Original Image", fontsize=14)
    axes[0].axis('off')

    # --- 두 번째: CLIP 바이너리 마스크 (NEW) ---
    # cmap='gray'를 사용하여 0은 검은색, 1은 흰색으로 명확하게 보여줍니다.
    # vmin=0, vmax=1로 설정하여 확실한 대비를 줍니다.
    axes[1].imshow(clip_heatmap)
    axes[1].set_title("2. CLIP", fontsize=14)
    axes[1].axis('off')

    # --- 세 번째: 어텐션 히트맵 (기존 axes[1]에서 이동) ---
    axes[2].imshow(heatmap_color)
    axes[2].set_title("3. Masked Attention Heatmap", fontsize=14)
    axes[2].axis('off')

    # --- 네 번째: 오버레이 (기존 axes[2]에서 이동) ---
    axes[3].imshow(overlay_img)
    axes[3].set_title("4. Final Overlay View", fontsize=14)
    axes[3].axis('off')

    axes[4].imshow(orig_img)
    axes[4].imshow(gt_map, cmap='jet', alpha=0.5)
    axes[4].set_title('GT')
    axes[4].axis('off')
    # 상단 여백 조정 (제목 공간 확보)
    plt.subplots_adjust(top=0.85)
    # 7. 파일 저장
    save_dir = "simple_top3_correl_selection_top3percent"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{object_name}_{action}_{filename}.png"
    
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"✅ 합산 이미지 저장 완료: {save_path}")