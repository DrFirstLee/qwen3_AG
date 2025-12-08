from config import AGD20K_PATH 
from PIL import Image
import io
import base64

import api_key
import openai
client_gpt = openai.OpenAI()
gpt_model_name="gpt-5.1-2025-11-13"
client_gemini = openai.OpenAI(
    api_key=api_key.gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
gemini_model_name = "gemini-2.5-pro"


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
        "reason": "<short explanation why>"
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


def get_df_from_logs(file_path):
    # 데이터를 담을 리스트
    data_list = []

    # 정규표현식 패턴 컴파일
    # 패턴 1: Action, Object, image_name 추출
    # 구조: Action : {값}, Object : {값} image_name : {값}
    pattern_meta = re.compile(r"Action\s*:\s*(.+?),\s*Object\s*:\s*(.+?)\s+image_name\s*:\s*(.+)")

    # 패턴 2: parsed dots 추출
    # 구조: parsed dots!!! : {리스트형태}
    pattern_dots = re.compile(r"parsed dots!!!\s*:\s*(.+)")

    # 임시 저장용 딕셔너리
    current_entry = {}



    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 1. Action, Object, Filename 라인 찾기
            match_meta = pattern_meta.search(line)
            if match_meta:
                # 새로운 항목 시작
                current_entry = {} 
                current_entry['Action'] = match_meta.group(1).strip()
                current_entry['Object'] = match_meta.group(2).strip()
                current_entry['filename'] = match_meta.group(3).strip()
                continue
            
            # 2. Parsed dots 라인 찾기
            match_dots = pattern_dots.search(line)
            if match_dots and 'Action' in current_entry:
                dots_str = match_dots.group(1).strip()
                try:
                    # 문자열 "[[1,2], ...]"을 실제 리스트 객체로 변환
                    current_entry['parsed_dots'] = ast.literal_eval(dots_str)
                except:
                    # 변환 실패 시 문자열 그대로 저장
                    current_entry['parsed_dots'] = dots_str
                
                # 필요한 정보가 다 모였으므로 리스트에 추가
                data_list.append(current_entry)
                current_entry = {} # 초기화

    # 데이터프레임 생성
    df = pd.DataFrame(data_list)
    df.columns = ['action','object','filename','dots']
    df_fin = df.loc[df[['action','object','filename']].drop_duplicates().index].reset_index(drop=True)
    # df = df.drop_duplicates()#.reset_index(drop=True)
    print(f">>>>> Total data Length : {len(df_fin)}")
    return df_fin


# 파일 경로 설정
file_path = '/home/bongo/porter_notebook/research/qwen3/32B_ego_exo_relative_prompt5/ego_exo_prompt5_relative.log'

df_fin = get_df_from_logs(file_path)

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

        question = "tell me about the image"
        messages = [
        {"role": "system", "content": system_prompt},
        {
        "role": "user",
        "content": [
            {"type": "text", "text": input_prompt(action, object_name, dot)},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
        }]
        # 2. 추론 (OpenAI API 호출)
        response = client_gemini.chat.completions.create(
        model= gemini_model_name,
        messages=messages,
        )
        result = response.choices[0].message.content
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

df_fin['veri_result'] = result_row
df_fin['veri_reason'] = reason_row
df_fin['final_dot'] = final_dot_row


df_fin.to_pickle('test_verify.pkl')

    