import torch
# import clip # OpenAI의 CLIP 라이브러리 (또는 CLIP Surgery 버전)
from PIL import Image
import numpy as np
import cv2 # CLIP Surgery는 CV2를 사용하는 것으로 보입니다.
import matplotlib.pyplot as plt

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
clip_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

from PIL import Image
import torch
import matplotlib.pyplot as plt

SOURCE_IMAGE_DIR = "/home/DATA/AGD20K/Seen/testset/egocentric"
image_path = f"{SOURCE_IMAGE_DIR}/hold/toothbrush/toothbrush_003341.jpg" #apple_001541.jpg"

original_image = Image.open(image_path).convert('RGB')

# 1. 단일 텍스트 프롬프트 정의
prompt_text = '''
Handle grip.
'''

# 2. 입력 처리
# processor는 텍스트 입력을 리스트 형태로 기대하므로 [prompt_text]로 전달합니다.
# 이미지도 단일 이미지를 리스트에 담아 [original_image]로 전달합니다.
inputs = processor(
    text=[prompt_text], 
    images=[original_image], 
    padding="max_length", 
    return_tensors="pt"
) # .to(device)는 inputs 딕셔너리에 적용

# 3. 예측 (기존 코드와 동일)
with torch.no_grad():
    outputs = clip_model(**inputs)
    # preds의 shape은 [1, 1, H, W]가 됩니다 (배치 크기 1, 프롬프트 1개)
    preds = outputs.logits.unsqueeze(0).unsqueeze(1) 

# 4. 시각화 (루프 제거)
# 원본 1개 + 히트맵 1개 = 총 2개의 플롯을 생성합니다.
_, ax = plt.subplots(1, 2, figsize=(8, 4)) # figsize을 1x2에 맞게 조정
# [a.axis('off') for a in ax.flatten()]

# 5. 원본 이미지 그리기 (ax[0])
ax[0].imshow(original_image)
ax[0].set_title("Original Image")

# 6. 히트맵 그리기 (ax[1])
# preds[0][0]는 첫 번째 (유일한) 예측 결과를 의미합니다.
ax[1].imshow(torch.sigmoid(preds[0][0]).cpu().detach().squeeze()) 
ax[1].set_title(f"Heatmap for '{prompt_text}'")

plt.tight_layout()

plt.savefig("anchor_sample_tooth_hold.jpg")
print("이미지가  저장되었습니다.")
plt.show()