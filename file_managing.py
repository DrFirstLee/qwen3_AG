import os
import json
from pathlib import Path
from config import AGD20K_PATH
from io import BytesIO
import base64
from PIL import Image

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

