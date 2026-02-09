import os
import json
import random
import pickle
from pathlib import Path
from tqdm import tqdm
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("/home/bongo/porter_notebook/research/qwen3") 


from config import AGD20K_PATH, model_name

# --- 경로 설정 (기존 코드와 동일하게 맞춤) ---

TARGET_JSON_PATH = "/home/bongo/porter_notebook/research/qwen3/selected_samples.json"
SAVE_CACHE_NAME = "fixed_exo_candidates.pkl"

# 1. JSON 로드
print(f"📂 Loading target JSON from {TARGET_JSON_PATH}")
with open(TARGET_JSON_PATH, 'r') as f:
    json_data = json.load(f)

# 2. Exo 이미지 샘플링 (Case ID를 키로 저장)
exo_cache = {}
EXO_ROOT_BASE = Path(f"{AGD20K_PATH}/Seen/trainset/exocentric")
valid_ext = {'.jpg', '.jpeg', '.png'}

print("🎲 Randomly selecting 20 exo images per case...")

for case_id, item in tqdm(json_data["selected_samples"].items()):
    action = item["action"]
    object_name = item["object"]

    
    # Exo 디렉토리 경로
    exo_dir = EXO_ROOT_BASE / action / object_name
    

    # 모든 이미지 파일 탐색
    all_exo_files = [p for p in exo_dir.rglob("*") if p.suffix.lower() in valid_ext]
    
    # 20개 랜덤 샘플링 (파일이 20개보다 적으면 전체 선택)
    if not all_exo_files:
        selected_files = []
    else:
        # 여기서 랜덤성이 발생하지만, 한 번 저장하면 고정됨
        selected_files = random.sample(all_exo_files, min(len(all_exo_files), 20))
        
    # 경로를 문자열로 변환하여 저장
    exo_cache[case_id] = [str(f) for f in selected_files]

# 3. 결과 저장
with open(SAVE_CACHE_NAME, 'wb') as f:
    pickle.dump(exo_cache, f)

print(f"✅ Saved {len(exo_cache)} cases to {SAVE_CACHE_NAME}")