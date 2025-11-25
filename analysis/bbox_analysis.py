
import json
import os
import ast
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# 1. 경로 설정 (환경에 맞게 수정해주세요)
# ==========================================
JSON_PATH = '/home/bongo/porter_notebook/research/qwen3/results_32B/results_vlm_bbox.json'

# [중요] 원본 이미지가 들어있는 폴더 경로를 입력하세요.
# 예: '/home/bongo/dataset/images/'
SOURCE_IMAGE_DIR = "/home/DATA/AGD20K/Seen/testset/egocentric"

OUTPUT_DIR = 'bbox'


# 저장할 폴더가 없으면 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# JSON 로드
with open(JSON_PATH, 'r', encoding='utf-16') as f:
    data = json.load(f)

print(f"총 {len(data)}개의 데이터를 처리합니다...")

for key_str, bbox_str in data.items():
    try:
        # 1. 키 파싱 (object_name$action$file_name)
        # 예: "apple$cut$apple_000054.jpg" -> ["apple", "cut", "apple_000054.jpg"]
        parts = key_str.split('$')
        
        if len(parts) < 3:
            print(f"[Skip] 형식이 맞지 않는 키: {key_str}")
            continue

        object_name = parts[0]
        action = parts[1]
        file_name = parts[2]  # 실제 이미지 파일명

        # 2. BBox 문자열 파싱 "[427, 372, 805, 887]" -> [427, 372, 805, 887]
        bbox = ast.literal_eval(bbox_str)

        # 3. 원본 이미지 불러오기
        img_full_path = os.path.join(SOURCE_IMAGE_DIR, action, object_name,file_name)
        
        if not os.path.exists(img_full_path):
            print(f"[Error] 이미지를 찾을 수 없음: {img_full_path}")
            continue

        with Image.open(img_full_path).convert("RGB") as img:
            draw = ImageDraw.Draw(img)

            # 4. BBox 그리기 (x1, y1, x2, y2)
            # outline: 테두리 색상, width: 두께
            draw.rectangle(bbox, outline="red", width=5)

            # (선택) 텍스트 추가: 어떤 액션인지 박스 위에 글씨 쓰기
            # 폰트가 없으면 기본 폰트 사용 (한글 깨질 수 있음, 영문 추천)
            try:
                # 리눅스 기본 폰트 경로 예시 (없으면 생략 가능)
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
            except:
                font = ImageFont.load_default()
            
            label_text = f"{object_name}: {action}"
            # 텍스트 배경 그리기 (가독성 위해)
            text_pos = (bbox[0], bbox[1] - 35 if bbox[1] > 35 else bbox[1])
            draw.text(text_pos, label_text, fill="red", font=font)

            # 5. 이미지 저장
            # 파일명 충돌 방지를 위해 key_str 전체를 파일명으로 사용하거나
            # file_name 앞에 action을 붙여서 저장
            save_name = f"{object_name}_{action}_{file_name}"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            
            img.save(save_path)
            print(f"[Saved] {save_path}")

    except Exception as e:
        print(f"[Fail] 처리 중 에러 발생 ({key_str}): {e}")

