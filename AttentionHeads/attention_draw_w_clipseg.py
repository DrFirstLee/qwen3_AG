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


df_attention = pd.read_pickle("attention_result_32B.pkl")
print(f"Length of dataset : {len(df_attention)}")
top_10_frequency_counter = Counter()

head_performance = {}

for index, row in df_attention.iterrows():
    object_name = row['object']
    action = row['action']
    filename = row['filename']
    description = row['description']
    attention_value = row['s_img']
    file_name_real = f"{AGD20K_PATH}/Seen/testset/egocentric/{action}/{object_name}/{filename}"
    print(f"Processing image {index}, object : {object_name}, action : {action}")

    clip_heatmap = get_clipseg_heatmap(
        file_name_real,
        clip_model, # Pass the model object (now on GPU)
        processor,
        object_name,
    )
    # 2. CLIPSeg 히트맵을 어텐션 맵 크기(31x31)로 리사이즈 (비교를 위해)
    clip_heatmap_resized = cv2.resize(clip_heatmap, (31, 31), interpolation=cv2.INTER_LINEAR)
    clip_flat = clip_heatmap_resized.flatten()
    current_image_scores = []

    for idx in attention_value: 
        layer = idx['layer']
        head = idx['head']
        if layer == 26 and head == 20:
            inside_heatmap = idx['heatmap']
            
            # 1. 원본 이미지 로드 (OpenCV는 BGR이므로 RGB로 변환)
            orig_img = cv2.imread(file_name_real)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            h, w, _ = orig_img.shape

            # 2. 어텐션 히트맵 전처리 및 리사이즈
            # 0~1 사이로 정규화 (이미 되어있을 수 있지만 안전을 위해)
            attn_norm = (inside_heatmap - inside_heatmap.min()) / (inside_heatmap.max() - inside_heatmap.min() + 1e-8)
            
            # 원본 이미지 크기로 인터폴레이션 (BILINEAR 추천)
            attn_resized = cv2.resize(attn_norm, (w, h), interpolation=cv2.INTER_LINEAR)

            # 3. 히트맵 컬러맵 적용 (JET 컬러맵 사용)
            heatmap_color = cv2.applyColorMap(np.uint8(255 * attn_resized), cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) # RGB 변환

            # 4. 합성 이미지 생성 (원본 60% + 히트맵 40%)
            overlay_img = cv2.addWeighted(orig_img, 0.6, heatmap_color, 0.4, 0)

            # 4. 시각화 설정 (figsize를 조금 키워 텍스트 공간 확보)
            fig, axes = plt.subplots(1, 3, figsize=(22, 10))
            
            # 상단 제목 정보 구성 (description은 100자마다 줄바꿈)
            wrapped_desc = "\n".join(textwrap.wrap(f"Description: {description}", width=120))
            full_title = (
                f"Object: {object_name}  |  Action: {action}  |  File: {filename}\n"
                f"{wrapped_desc}"
            )
            
            # 전체 제목 추가 (폰트 크기 및 위치 조정)
            plt.suptitle(full_title, fontsize=15, fontweight='bold', y=0.95)

            # 각 서브플롯 표시
            axes[0].imshow(orig_img)
            axes[0].set_title("Original Image", fontsize=12)
            axes[0].axis('off')

            axes[1].imshow(heatmap_color)
            axes[1].set_title(f"Attention Heatmap (Layer {layer} Head {head})", fontsize=12)
            axes[1].axis('off')

            axes[2].imshow(overlay_img)
            axes[2].set_title("Overlay (Attention + Image)", fontsize=12)
            axes[2].axis('off')

            # 5. 여백 조정 및 저장
            plt.subplots_adjust(top=0.82) # 제목이 이미지와 겹치지 않게 상단 여백 조절
            
            save_dir = "vis_results_L26_H20"
            os.makedirs(save_dir, exist_ok=True)
            save_path = f"{save_dir}/{object_name}_{action}_{filename}.png"
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()

            print(f"📸 시각화 완료: {save_path}")