import os
import json
from pathlib import Path
from config import AGD20K_PATH
from io import BytesIO
import base64
from PIL import Image
import torch
import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import transforms


clip_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
clip_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")



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
        clip_processor, 
        object_name: str,
    ):
    """
    (수정됨) CLIPSeg 모델을 사용하여 이미지와 텍스트 프롬프트 간의
    세그멘테이션 히트맵을 추출합니다.
    """
    if model is None or clip_processor is None:
        print("Error: CLIPSeg model or processor not loaded.")
        return None, None
    
    original_image = Image.open(image_path).convert('RGB')
    original_size = original_image.size # (width, height)

    # 1. 단일 텍스트 프롬프트 정의
    prompt_text = object_name

    # 2. 입력 처리
    inputs = clip_processor(
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


def load_selected_samples(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_actual_path(path_with_variable):
    """Convert ${AGD20K_PATH} to actual path"""
    return path_with_variable.replace("${AGD20K_PATH}", AGD20K_PATH)

def make_input_image(file_name_real):
    # 1. 이미지 열기 및 리사이즈
    with Image.open(file_name_real) as img:
        img = img.convert("RGB")
        resized_image = img.resize((1000, 1000))
        
        # 2. 함수 내부에서 버퍼 생성 (with 구문 사용 추천 X -> getvalue 후엔 자동 GC됨)
        buffered = BytesIO()
        # 3. 버퍼에 저장 (메모리에 JPEG 생성)
        resized_image.save(buffered, format="JPEG")
        
        # 4. 바로 인코딩 후 리턴 (한 줄로 처리)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


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
    
prompt_dict_obj = {
    "beat": {
        "drum": "drum",
    },
    "boxing": {
        "punching_bag": "punching bag",
    },
    "brush_with": {
        "toothbrush": "toothbrush",
    },
    "carry": {
        "skateboard": "skateboard",
        "skis": "skis",
        "snowboard": "snowboard",
        "surfboard": "surfboard",
    },
    "catch": {
        "frisbee": "frisbee",
        "rugby_ball": "rugby ball",
        "soccer_ball": "soccer ball",
    },
    "cut": {
        "apple": "apple",
        "banana": "banana",
        "carrot": "carrot",
        "orange": "orange",
    },
    "cut_with": {
        "knife": "knife blade",
        "scissors": "scissors blade",
    },
    "drag": {
        "suitcase": "suitcase",
    },
    "drink_with": {
        "bottle": "bottle cap",
        "cup": "cup",
        "wine_glass": "wineglass",
    },
    "eat": {
        "apple": "fruit",
        "banana": "banana",
        "broccoli": "broccoli",
        "carrot": "carrot",
        "hot_dog": "hot dog",
        "orange": "orange",
    },
    "hit": {
        "axe": "axe handle",
        "baseball_bat": "baseball bat",
        "hammer": "hammer handle",
        "tennis_racket": "tennis racket handle",
    },
    "hold": {
        "axe": "axe handle",
        "badminton_racket": "badminton racket handle",
        "baseball_bat": "baseball bat handle",
        "book": "book page",
        "bottle": "bottle body",
        "bowl": "bowl",
        "cup": "cup handle",
        "fork": "fork handle",
        "frisbee": "frisbee",
        "golf_clubs": "golf club handle",
        "hammer": "hammer handle",
        "knife": "knife handle",
        "scissors": "scissors handle",
        "skateboard": "skateboard",
        "skis": "skis",
        "snowboard": "snowboard",
        "suitcase": "suitcase",
        "surfboard": "surfboard",
        "tennis_racket": "tennis racket handle",
        "toothbrush": "toothbrush handle",
        "wine_glass": "wineglass neck"
    },
    "jump":{
        "skateboard": "skateboard",
        "skis": "skis",
        "snowboard": "snowboard",
        "surfboard": "surfboard",
    },
    "kick": {
        "punching_bag": "punching bag",
        "rugby_ball": "rugby ball",
        "soccer_ball": "soccer ball"
    },
    "lie_on": {
        "bed": "bed",
        "bench": "bench seat",
        "couch": "couch seat",
        "surfboard": "surfboard"
    },
    "lift":{
        "fork": "fork handle",
    },
    "look_out":{
        "binoculars": "binoculars"
    },
    "open": {
        "book": "book page",
        "bottle": "bottle cap",
        "microwave": "microwave door handle",
        "oven": "oven door handle",
        "refrigerator": "refrigerator door handle",
        "suitcase": "suitcase",
    },
    "pack": {
        "suitcase": "suitcase",
    },
    "peel": {
        "apple": "fruit",
        "banana": "banana",
        "carrot": "carrot",
        "orange": "orange",
    },
    "pick_up": {
        "suitcase": "suitcase",
        "skis": "skis",
    },
    "pour": {
        "bottle": "bottle body",
        "cup": "cup handle",
        "wine_glass": "wine glass neck",
    },
    "push": {
        "bicycle": "bicycle handlebars",
        "motorcycle": "motorcycle handlebars.motorcycle seat",
    },
    "ride": {
        "bicycle": "bicycle handlebars.bicycle pedal.bicycle seat",
        "motorcycle": "motorcycle handlebars.motorcycle seat.motorcycle footrest",
    },
    "sip": {
        "bottle": "bottle cap",
        "cup": "cup",
        "wine_glass": "wine glass",
    },
    "sit_on": {
        "bed": "bed",
        "bench": "bench seat",
        "bicycle": "bicycle seat",
        "chair": "chair seat",
        "couch": "couch seat",
        "motorcycle": "motorcycle seat",
        "skateboard": "skateboard top",
        "surfboard": "surfboard",
    },
    "stick": {
        "fork": "fork tines",
        "knife": "knife blade",
    },
    "stir": {
        "bowl": "bowl inside",
    },
    "swing": {
        "badminton_racket": "badminton racket handle",
        "baseball_bat": "baseball bat handle",
        "golf_clubs": "golf club handle",
        "tennis_racket": "tennis racket handle",
    },
    "take_photo": {
        "camera": "camera grip",
        "cell_phone": "cell phone",
    },
    "talk_on": {
        "cell_phone": "cell phone screen",
    },
    "text_on": {
        "cell_phone": "cell phone screen",
    },
    "talk_on": {
        "cell_phone": "cell phone screen",
    },
    "throw": {
        "baseball": "baseball",
        "basketball": "basketball",
        "discus": "discus",
        "frisbee": "frisbee",
        "javelin": "javelin handle",
        "rugby_ball": "rugby ball",
        "soccer_ball": "soccer ball",
    },
    "type_on": {
        "keyboard": "keyboard",
        "laptop": "laptop keyboard",
    },
    "wash": {
        "bowl": "bowl",
        "carrot": "carrot",
        "cup": "cup",
        "fork": "fork tines",
        "knife": "knife blade",
        "orange": "orange",
        "toothbrush": "toothbrush",
        "wine_glass": "wine glass body",
    },
    "write": {
        "pen": "pen grip",
    },
}

