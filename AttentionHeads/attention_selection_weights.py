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
import pickle

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

def calculate_attention_ratio(heatmap_31x31, binary_mask_original):
    """
    (마스크 내부 어텐션 합) / (전체 어텐션 합) 비율을 계산합니다.
    """
    # 1. 바이너리 마스크를 어텐션 크기(31x31)로 리사이즈
    # 인터폴레이션은 INTER_NEAREST를 써서 0과 1의 경계를 유지하는 것이 정확합니다.
    mask_resized = cv2.resize(binary_mask_original, (31, 31), interpolation=cv2.INTER_NEAREST)
    
    # 2. 마스크 내부 어텐션 합 계산
    # 두 행렬을 곱하면 마스크가 0인 부분의 어텐션은 모두 0이 됩니다.
    inside_sum = np.sum(heatmap_31x31 * mask_resized)
    
    # 3. 전체 어텐션 합 계산
    total_sum = np.sum(heatmap_31x31)
    
    # 4. 비율 계산 (0으로 나누기 방지)
    ratio = (inside_sum / total_sum) if total_sum > 0 else 0
    return ratio


df_attention = pd.read_pickle("attention_result_32B.pkl")
print(f"Length of dataset : {len(df_attention)}")
top_10_frequency_counter = Counter()

head_performance = {}
all_image_scores = []

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
    clip_binary_mask = (clip_heatmap > 0.15).astype(np.float32)
    
    # 2. CLIPSeg resize as same as Qwen3 path  size (31x31)
    clip_heatmap_resized = cv2.resize(clip_heatmap, (31, 31), interpolation=cv2.INTER_LINEAR)
    # clip_heatmap_resized = cv2.resize(clip_binary_mask, (31, 31), interpolation=cv2.INTER_LINEAR)
    
    clip_flat = clip_heatmap_resized.flatten()
    
    

    for idx in attention_value: 
        layer = idx['layer']
        head = idx['head']
        inside_heatmap = idx['heatmap']
        inside_flat = inside_heatmap.flatten()
        # print(f"layer : {idx['layer']}, head : {idx['head']} , S_img : {idx['S_img']}")

        score, _ = pearsonr(inside_flat, clip_flat)
        

        threshold = np.percentile(inside_flat, 97)
        filtered_heatmap = np.where(inside_heatmap >= threshold, inside_heatmap, 0)
        filtered_inside_flat = filtered_heatmap.flatten()

        # score = calculate_attention_ratio(inside_heatmap, clip_binary_mask)
        # score = calculate_attention_ratio(filtered_heatmap, clip_binary_mask)
        score, _ = pearsonr(filtered_inside_flat, clip_flat)

        all_image_scores.append({
            'object': object_name,
            'action': action,
            'filename': filename,
            'layer': layer,
            'head': head,
            'score': score
        })
        # break


df = pd.DataFrame(all_image_scores)
# 루프가 완전히 종료된 후 한 번에 저장
save_file = "all_input_attention_scores_correl.pkl"
df.to_pickle(save_file)
print(f"전체 {len(all_image_scores)}개의 이미지 점수가 {save_file}에 저장되었습니다.")