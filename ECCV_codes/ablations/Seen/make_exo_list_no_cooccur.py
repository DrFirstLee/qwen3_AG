import os
import json
import random
import pickle
import sys
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# --- 경로 설정 ---
# (사용자 환경에 맞게 경로가 잡혀있다고 가정합니다)
sys.path.append("/home/bongo/porter_notebook/research/qwen3") 
from config import AGD20K_PATH, model_name

TARGET_JSON_PATH = "/home/bongo/porter_notebook/research/qwen3/selected_samples.json"
SAVE_CACHE_NAME = "unique_first_exo_candidates.pkl" # 파일명 변경

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
# 2. [Pre-computation] 전체 Exo 데이터셋 스캔하여 Unique 여부 파악
# ---------------------------------------------------------
print("🔍 Scanning ALL exocentric images to identify uniqueness...")

# global_id_map: 이미지ID -> {(action, object), ...} 집합
global_id_map = defaultdict(set)

# 전체 디렉토리 순회
# (주의: JSON에 없는 케이스까지 포함하여 전체를 봐야 진정한 Unique를 알 수 있음)
for action_dir in EXO_ROOT_BASE.iterdir():
    if not action_dir.is_dir(): continue
    
    action = action_dir.name
    for obj_dir in action_dir.iterdir():
        if not obj_dir.is_dir(): continue
        
        obj = obj_dir.name
        
        # 파일 순회
        for img_path in obj_dir.glob("*"):
            if img_path.suffix.lower() in valid_ext:
                img_id = get_image_id(img_path.name)
                global_id_map[img_id].add((action, obj))

print(f"✅ Global scan complete. Found {len(global_id_map)} unique image IDs.")

# ---------------------------------------------------------
# 3. Main Selection Loop (Unique 우선 랜덤 샘플링)
# ---------------------------------------------------------
print(f"📂 Loading target JSON from {TARGET_JSON_PATH}")
with open(TARGET_JSON_PATH, 'r') as f:
    json_data = json.load(f)

exo_cache = {}
TARGET_COUNT = 1

print("🎲 Selecting images (Priority: Unique > Overlap)...")

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

    # 2. Unique와 Overlap으로 분류
    unique_candidates = []
    overlap_candidates = []
    
    for f_path in all_exo_files:
        img_id = get_image_id(f_path.name)
        
        # global_id_map에서 해당 ID를 가진 (action, object) 쌍이 1개뿐이면 Unique
        if len(global_id_map[img_id]) == 1:
            unique_candidates.append(f_path)
        else:
            overlap_candidates.append(f_path)
            
    # 3. 우선순위 샘플링 로직
    # (A) Unique 후보들 셔플
    random.shuffle(unique_candidates)
    
    # (B) Overlap 후보들 셔플
    random.shuffle(overlap_candidates)
    
    selected_files = []
    
    # Case 1: Unique 만으로 20개 이상인 경우 -> Unique에서만 뽑음
    if len(unique_candidates) >= TARGET_COUNT:
        selected_files = unique_candidates[:TARGET_COUNT]
        
    # Case 2: Unique가 부족한 경우 -> Unique 다 넣고, 나머지를 Overlap에서 충원
    else:
        selected_files.extend(unique_candidates) # 일단 다 넣음
        
        remainder = TARGET_COUNT - len(selected_files)
        # Overlap에서 남은 개수만큼 가져오기 (Overlap이 부족하면 있는 만큼만)
        selected_files.extend(overlap_candidates[:remainder])
    
    # 4. 결과 저장 (문자열 변환)
    exo_cache[case_id] = [str(f) for f in selected_files]

# ---------------------------------------------------------
# 4. 결과 파일 저장
# ---------------------------------------------------------
with open(SAVE_CACHE_NAME, 'wb') as f:
    pickle.dump(exo_cache, f)

print(f"✅ Saved {len(exo_cache)} cases to {SAVE_CACHE_NAME}")
print("   Strategy: Filled with Unique first, then Overlap.")