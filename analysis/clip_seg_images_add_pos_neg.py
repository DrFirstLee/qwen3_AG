import os
import sys

# ------------------------------------------------------
# 1. Third-Party Libraries (외부 라이브러리)
# ------------------------------------------------------
import cv2
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import transforms
from PIL import Image
from sklearn.cluster import KMeans
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# ------------------------------------------------------
# 2. System Path Setup (로컬 모듈 경로 설정)
# ------------------------------------------------------
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
)

# ------------------------------------------------------
# 4. Model Initialization (모델 및 설정 로드)
# ------------------------------------------------------
# CLIPSeg 모델 로드
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
clip_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

print("Imports and Model Loading Completed.")

def load_vlm_heatmap(heatmap_path: str, target_size: tuple) -> np.ndarray:
    """저장된 VLM 히트맵을 불러옵니다."""
    heatmap_img = Image.open(heatmap_path).convert('L')
    heatmap_img = heatmap_img.resize(target_size, resample=Image.Resampling.BILINEAR)
    heatmap_array = np.array(heatmap_img).astype(np.float32) / 255.0
    return heatmap_array




def load_ground_truth( gt_path):
    """
    Load and process ground truth image
    Args:
        gt_path (str): Path to the ground truth image
    Returns:
        torch.Tensor: Processed ground truth tensor normalized to [0, 1]
    """
    try:
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
    except Exception as e:
        print(f"⚠️ Failed to load ground truth image: {str(e)}")
        return None    

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
        final_heatmap = final_heatmap ** gamma ### + epsilon
        
    return final_heatmap

#### PART INFO FROM QWEN3 -VL-32B
import json
json_dir = "/home/bongo/porter_notebook/research/qwen3/32B_ask_vlms/32B_vlm_parts_answer.json"
# 1. 파일 읽기 (1차 파싱: 키는 문자열, 값은 '문자열로 된 리스트' 상태)
with open(json_dir, 'r', encoding='utf-16') as f:
    raw_data = json.load(f)

# 2. 값(Value)을 실제 리스트/딕셔너리로 변환 (2차 파싱)
result_dict = {}

for key, value_str in raw_data.items():

    # 문자열로 되어 있는 리스트("[\n{...}]")를 실제 파이썬 리스트로 변환
    parsed_value = json.loads(value_str)
    result_dict[key] = parsed_value

result_dict
##

gamma = 0.5
epsilon = 0.1#1e-6 #
metrics_tracker_dino = MetricsTracker(name="only_ego")

json_path = os.path.join("/home/bongo/porter_notebook/research/qwen3","selected_samples.json")
data = load_selected_samples(json_path)
missing_gt = 0
processed_count = 0

# Get total number of samples
total_samples = len(data['selected_samples'])


# Process each sample
print(f"Processing {total_samples} samples...")
print("=" * 50)    
for pair_key, sample_info in data["selected_samples"].items():
    processed_count += 1
    print(f"--- Start  {processed_count}  / {total_samples}", "-"*80) 
    original_image_path = sample_info['image_path'].replace("${AGD20K_PATH}",'/home/DATA/AGD20K')
    
    file_name = os.path.basename(sample_info['image_path'])
    action_name = sample_info['image_path'].split('/')[4]
    if file_name.count("_") ==1:
        item_name = file_name.split("_")[0]
    else:
        item_name = file_name.split("_")[0] + "_" + file_name.split("_")[1]
    AGD20K_PATH = '/home/DATA/AGD20K'
    vlm_heatmap_path = f"/home/bongo/porter_notebook/research/qwen3/32B_ego_exo_relative_prompt5/heatmaps/{file_name.split('.')[0]}_{action_name}_heatmap.jpg"
    gt_path =  f"{AGD20K_PATH}/Seen/testset/GT/{action_name}/{item_name}/{file_name.split('.')[0]}.png"
    dot_path = f"/home/bongo/porter_notebook/research/qwen3/32B_ego_exo_relative_prompt5/dots_only/{file_name.split('.')[0]}_{action_name}_dots.jpg"
    print(item_name, action_name, file_name)
    output_path = f"clipseg_32B_prompt5_pos_only/{file_name.split('.')[0]}_{action_name}.png"
    # --- 2. VLM 히트맵 로드 및 DINO 특징 추출 ---
    original_image = Image.open(original_image_path).convert('RGB')
    try:
        dot_image = Image.open(dot_path).convert('RGB')
    except:
        print(f" NO Image!! {file_name.split('.')[0]}_{action_name}")
        continue
    vlm_heatmap_image =  Image.open(vlm_heatmap_path).convert('RGB')
#     print("Loading VLM heatmap...")
    vlm_heatmap = load_vlm_heatmap(vlm_heatmap_path, original_image.size)


    clip_heatmap = get_clipseg_heatmap(
        original_image_path,
        clip_model, # Pass the model object (now on GPU)
        processor,
        action_name + " " + item_name,
    )
    ## ADDED
    current_key = f"{item_name}_{action_name}"
    object_mask = (clip_heatmap > 0).astype(float)


    if current_key in result_dict:
        # 해당 키에 대한 부품 리스트 가져오기
        parts_list = result_dict[current_key]
        
        print(f"Applying parts refinement for {current_key}: {len(parts_list)} parts")
        
        for part_info in parts_list:
            p_name = part_info['part']   # 예: "handle"
            p_label = part_info['label'] # 예: "positive" or "negative"
            
            # 3. 부품(Part)에 대한 CLIPSeg 히트맵 추출
            part_heatmap = get_clipseg_heatmap(
                image_path=original_image_path,
                model=clip_model,     # 루프 밖에서 정의된 모델
                processor=processor,  # 루프 밖에서 정의된 프로세서
                object_name=p_name    # 객체 전체가 아닌 '부품 이름'으로 추출
            )
            
            if part_heatmap is not None:
                # 4. 마스킹 적용: 원본 객체가 있는 위치에서만 부품 값을 유효하게 함
                # (배경에 있는 다른 물체의 'handle' 등이 잡히는 것을 방지)
                masked_part_heatmap = part_heatmap * object_mask
                print(f"{item_name} // p_name : {p_name} , p_label : {p_label}")
                # 5. Positive / Negative 적용
                # 가중치(weight)를 0.5~1.0 정도로 조절하여 반영 강도를 튜닝할 수도 있습니다.
                refinement_strength = 1.0 
                
                if p_label == "positive":
                    clip_heatmap = (masked_part_heatmap * refinement_strength)
                # elif p_label == "negative":
                #     clip_heatmap = clip_heatmap - (masked_part_heatmap * refinement_strength)

                # if p_label == "negative":
                #     clip_heatmap = clip_heatmap - (masked_part_heatmap * refinement_strength)

        # 6. 최종 클리핑 (MINMAX)
        clip_heatmap = (clip_heatmap - clip_heatmap.min()) / (clip_heatmap.max() - clip_heatmap.min())
        
    else:
        print(f"Warning: Key '{current_key}' not found in result_dict.")


    vlm_heatmap = vlm_heatmap ** gamma + epsilon
    vlm_fused_heatmap = vlm_heatmap * clip_heatmap
    # vlm_fused_heatmap = clip_heatmap

    
    # Calculate metrics if GT is available
    metrics = None
    gt_map = load_ground_truth(gt_path)
    if gt_map is not None:
        metrics_dino  = calculate_metrics(vlm_fused_heatmap, gt_map)
        metrics_tracker_dino.update(metrics_dino)
    else:
        print("NO GT!!!")
        continue
    metrics_tracker_dino.print_metrics(metrics_dino, vlm_heatmap_path.split('/')[-1])
    
    # --- 4. 결과 시각화 ---
    # ✨ 레이아웃을 1x4에서 1x5로 변경하고, figsize을 조정합니다.
    fig, ax = plt.subplots(1, 6, figsize=(25, 5))

    # --- Plot 1: 원본 이미지 (ax[0]) ---
    ax[0].imshow(original_image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # --- ✨ Plot 5: 최종 퓨전 히트맵 (기존 ax[3] -> ax[4]로 이동) ---
    ax[1].imshow(dot_image)
    ax[1].set_title('Dot image')
    ax[1].axis('off')
    
    # --- Plot 2: dot 히트맵 (ax[1]) ---
    ax[2].imshow(original_image)
    ax[2].imshow(vlm_heatmap_image , cmap='jet', alpha=0.5)
    ax[2].set_title('dot (Input)')
    ax[2].axis('off')

    # --- Plot 2: VLM 히트맵 (ax[1]) ---
    ax[3].imshow(original_image)
    ax[3].imshow(vlm_fused_heatmap, cmap='jet', alpha=0.5)
    ax[3].set_title('VLM Heatmap (Input)')
    ax[3].axis('off')
    
    # --- ✨ Plot 3: DINO 원본 히트맵 (새로 추가된 부분) ---
    # 이 dino_attention_heatmap 변수는 클러스터링 전에 미리 계산해 두어야 합니다.
    # (예: dino_attention_heatmap = generate_dino_heatmap(original_image_path, dino_model) )
    ax[4].imshow(original_image)
    ax[4].imshow(clip_heatmap, cmap='jet', alpha=0.5)
    ax[4].set_title('clip_heatmap')
    ax[4].axis('off')


    # --- ✨ Plot 5: 최종 퓨전 히트맵 (기존 ax[3] -> ax[4]로 이동) ---
    ax[5].imshow(original_image)
    ax[5].imshow(gt_map, cmap='jet', alpha=0.5)
    ax[5].set_title('GT')
    ax[5].axis('off')
#     전체 레이아웃 정리 및 출력
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

