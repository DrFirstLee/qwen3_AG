import os
import json
import random
import pickle
import sys
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# --- 경로 설정 ---
sys.path.append("/home/bongo/porter_notebook/research/qwen3") 
from config import AGD20K_PATH

TARGET_JSON_PATH = "/home/bongo/porter_notebook/research/qwen3/selected_samples.json"
SAVE_PKL_NAME = "random20_unique_candidates.pkl"  # 저장 파일명

EXO_ROOT_BASE = Path(f"{AGD20K_PATH}/Seen/trainset/exocentric")
valid_ext = {'.jpg', '.jpeg', '.png'}

# ---------------------------------------------------------
# 1. Helper Function: 이미지 ID 추출
# ---------------------------------------------------------
def get_image_id(filename):
    """
    파일명에서 고유 ID 추출 (action_object_XXXXXX.jpg -> XXXXXX.jpg)
    """
    return filename.split('_')[-1]

# ---------------------------------------------------------
# 2. [Pre-computation] 전체 Exo 데이터셋 스캔하여 Unique/Overlap 파악
# ---------------------------------------------------------
print("🔍 Scanning ALL exocentric images to identify uniqueness/overlap...")

global_id_map = defaultdict(set)

if not EXO_ROOT_BASE.exists():
    print(f"❌ Error: {EXO_ROOT_BASE} does not exist.")
    exit()

# 전체 디렉토리 순회
for action_dir in EXO_ROOT_BASE.iterdir():
    if not action_dir.is_dir(): continue
    action_name = action_dir.name
    
    for obj_dir in action_dir.iterdir():
        if not obj_dir.is_dir(): continue
        obj_name = obj_dir.name
        
        for img_path in obj_dir.glob("*"):
            if img_path.suffix.lower() in valid_ext:
                img_id = get_image_id(img_path.name)
                global_id_map[img_id].add((action_name, obj_name))

print(f"✅ Global scan complete. Mapped {len(global_id_map)} image IDs.")

# ---------------------------------------------------------
# 3. Main Selection Loop
# ---------------------------------------------------------
print(f"📂 Loading target JSON from {TARGET_JSON_PATH}")
with open(TARGET_JSON_PATH, 'r') as f:
    json_data = json.load(f)

exo_cache = {}
TARGET_COUNT = 20

print(f"🎲 Selecting up to {TARGET_COUNT} unique images per case...")

for case_id, item in tqdm(json_data["selected_samples"].items()):
    action = item["action"]
    object_name = item["object"]
    
    # 해당 케이스의 Exo 디렉토리
    exo_dir = EXO_ROOT_BASE / action / object_name
    
    # 1. 모든 후보 이미지 수집
    all_exo_files = [p for p in exo_dir.rglob("*") if p.suffix.lower() in valid_ext]
    
    if not all_exo_files:
        exo_cache[case_id] = []
        continue

    # 2. Unique 후보 분류
    unique_candidates = []
    
    for f_path in all_exo_files:
        img_id = get_image_id(f_path.name)
        
        # global_id_map에서 해당 ID를 가진 (action, object) 쌍이 1개뿐이면 Unique
        if len(global_id_map[img_id]) == 1:
            unique_candidates.append(f_path)
            
    # 3. 선택 로직
    selected_files = []
    
    # (A) Unique한 사진이 있는 경우 -> Unique에서만 선택
    if len(unique_candidates) > 0:
        if len(unique_candidates) > TARGET_COUNT:
            selected_files = random.sample(unique_candidates, TARGET_COUNT)
        else:
            selected_files = unique_candidates # 5개면 5개 전부
            
    # (B) Unique한 사진이 없는 경우 (0개) -> 전체에서 랜덤 선택
    else:
        if len(all_exo_files) > TARGET_COUNT:
            selected_files = random.sample(all_exo_files, TARGET_COUNT)
        else:
            selected_files = all_exo_files
    
    # 4. 결과 저장 (문자열 변환)
    exo_cache[case_id] = [str(f) for f in selected_files]

# ---------------------------------------------------------
# 4. 결과 파일 저장
# ---------------------------------------------------------
with open(SAVE_PKL_NAME, 'wb') as f:
    pickle.dump(exo_cache, f)

print(f"✅ Saved {len(exo_cache)} cases to {SAVE_PKL_NAME}")
# print("   Strategy: Only Unique (up to 20), if none -> Random 20.")
