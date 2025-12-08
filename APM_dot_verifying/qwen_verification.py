
from PIL import Image
import io
import sys
import base64

# nohup python -u ego_only.py > GPT5_relative_coord.log 2>&1 & tail -f GPT5_relative_coord.log
import os
import torch
import random
import json
from PIL import Image
# ------------------------------------------------------
# 2. System Path Setup (로컬 모듈 경로 설정)
# ------------------------------------------------------
# 현재 파일 위치 기준 상위 폴더를 시스템 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from file_managing import (
    load_selected_samples,
    get_actual_path,
    get_gt_path,
)
from config import AGD20K_PATH, model_name
from VLM_model_dot_relative import QwenVLModel, MetricsTracker

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ENABLE_SDPA"] = "1"


import re
import pandas as pd
import ast  # 문자열로 된 리스트를 실제 리스트로 변환하기 위해 사용

import json
import re


from io import BytesIO
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

# 2. 강력한 프롬프트 작성
system_prompt = """
You are an expert in Visual Affordance Grounding. 
Your task is to evaluate whether a specific pixel coordinate on an image is a valid region for a human to perform a specific action on an object.
"""


def input_prompt(action, object_name, dot):
    return f"""
    Analyze the provided image with the following details:

    1. **Target Action**: {action}
    2. **Target Object**: {object_name}
    3. **Query Point**: ({dot[0]},{dot[1]}) 
    4. **Image Resolution**: 1000x1000

    **Task**:
    Evaluate if the "Query Point" falls within the **affordance region** specific to the "{action}" on the "{object_name}". 
    (e.g., If action is 'jump' on 'skateboard', the point should be on the deck where feet act, not on the wheels or background.)
    **Output Format**:
    Provide the result in JSON format only:
    {{
        "result": <Pass or Fail>,
        "reason": "<in one sentence>"
    }}
    """



def parse_llm_json(text):
    """
    마크다운 코드 블록(```json ... ```)을 제거하고 JSON으로 변환하는 함수
    """
    try:
        # 1. 정규표현식으로 ```json 과 ``` 사이의 내용만 추출
        # re.DOTALL: 줄바꿈(\n)도 포함해서 찾기 위함
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        
        if match:
            json_str = match.group(1)  # 코드 블록 안의 내용만 가져옴
        else:
            json_str = text  # 코드 블록이 없으면 원본 그대로 사용 시도
            
        # 2. JSON 파싱
        return json.loads(json_str)
        
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 실패: {e}")
        return None
import pandas as pd
import ast
import re

def parse_log_to_df(file_path):
    data_list = []
    
    # 현재 처리 중인 샘플의 메타데이터 임시 저장용 (action, object, filename)
    current_meta = None 
    is_ego_section = False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # 1. Action 라인 파싱 (새로운 Ego 샘플 시작)
                # 포맷: Action : jump, Object : skis image_name : skis_002829.jpg
                if line.startswith("Action :"):
                    # 정규표현식으로 action, object, filename 추출
                    match = re.search(r"Action\s*:\s*(.*?),\s*Object\s*:\s*(.*?)\s+image_name\s*:\s*(.*)", line)
                    if match:
                        action = match.group(1).strip()
                        obj = match.group(2).strip()
                        filename = match.group(3).strip()
                        
                        current_meta = (action, obj, filename)
                        is_ego_section = True
                    continue

                # 2. Exo 라인 감지 (이 이후의 dots는 무시)
                if line.startswith("exo file name :"):
                    is_ego_section = False
                    continue

                # 3. Dots 파싱 및 데이터 병합
                if line.startswith("parsed dots!!! :"):
                    # Ego 섹션이고, 메타데이터가 확보된 상태일 때만 저장
                    if is_ego_section and current_meta is not None:
                        try:
                            dots_str = line.split(":", 1)[1].strip()
                            dots = ast.literal_eval(dots_str)
                            
                            # [action, object, filename, dots] 형태로 추가
                            data_list.append([current_meta[0], current_meta[1], current_meta[2], dots])
                            
                        except (ValueError, SyntaxError) as e:
                            print(f"Dots parsing error: {e} in line: {line}")
                            
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return pd.DataFrame()

    # --- DataFrame 생성 및 중복 처리 (요청하신 로직) ---
    df = pd.DataFrame(data_list)
    
    if not df.empty:
        df.columns = ['action', 'object', 'filename', 'dots']
        
        # action, object, filename 조합이 중복되는 경우 제거
        df_fin = df.loc[df[['action', 'object', 'filename']].drop_duplicates().index].reset_index(drop=True)
    else:
        # 데이터가 없을 경우 빈 DF 반환
        df_fin = pd.DataFrame(columns=['action', 'object', 'filename', 'dots'])

    print(f">>>>> Total data Length : {len(df_fin)}")
    return df_fin



model = QwenVLModel(model_name = model_name)
cnt_d = 0
# --- 실행 ---
file_path = '/home/bongo/porter_notebook/research/qwen3/32B_ego_exo_relative_prompt5/ego_exo_prompt5_relative.log'
df_fin = parse_log_to_df(file_path).head(4)
df_fin

print(f"length of Data : {len(df_fin)}")

result_row = []
reason_row = []
final_dot_row = []
for index, row in df_fin.iterrows():
    object_name = row['object']
    action = row['action']
    filename = row['filename']
    dot_list =  row['dots']
    file_name_real = f"{AGD20K_PATH}/Seen/testset/egocentric/{action}/{object_name}/{filename}"
    # if (object_name=='cup')&(action =='drink_with'):
    print(index,object_name,action,filename)
    image_base64 = make_input_image(file_name_real)
    dot_res_list = []
    dot_reason_list = []
    dot_real_list = []

    for dot in dot_list:
        messages = [
            
        {
        "role": "system", 
        "content": [
            {"type": "text", "text": system_prompt}
        ]
        },
        {
        "role": "user",
        "content": [
            {"type": "text", "text": input_prompt(action, object_name, dot)},
            {"type": "image", "image": f"data:image/jpeg;base64,{image_base64}"}
                    ]
        }
        ]
        # 2. 추론 (OpenAI API 호출)
        result = model.ask_with_messages(messages)
        result
        print(f"{dot} : {result}")
        llM_result_json = parse_llm_json(result)
        dot_res_list.append(llM_result_json['result'])
        dot_reason_list.append(llM_result_json['reason'])
        if llM_result_json['result']=='Pass':
            dot_real_list.append(dot)

        
    result_row.append(dot_res_list)
    reason_row.append(dot_reason_list)
    final_dot_row.append(dot_real_list)
    if cnt_d ==3 : 
        break
    cnt_d += 1
df_fin['veri_result'] = result_row
df_fin['veri_reason'] = reason_row
df_fin['final_dot'] = final_dot_row


df_fin.to_pickle('test_verify_qwen3_2b.pkl')

    