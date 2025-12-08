
from PIL import Image
import io
import sys
import base64

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# (루프 내부라고 가정)
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
# nohup python -u ego_only.py > GPT5_relative_coord.log 2>&1 & tail -f GPT5_relative_coord.log
import os
import torch
import random
import json
import numpy as np
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


print(f"🤖 {model_name} 모델 로딩중...")
# 1. Processor 로드

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
# 2. Qwen3 Model 로드
# 사용자가 제공한 코드에 맞춰 Qwen3VLForConditionalGeneration 사용
# dtype="auto", device_map="auto" 적용
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
    attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(model_name)
# 모델이 로드된 주 디바이스 확인 (DINO를 같은 곳에 올리기 위해)
device = model.device




cnt_d = 0
# --- 실행 ---
file_path = '/home/bongo/porter_notebook/research/qwen3/32B_ego_exo_relative_prompt5/ego_exo_prompt5_relative.log'
df_fin = parse_log_to_df(file_path).head(1)
df_fin
threshold_ratio = 0.5
print(f"length of Data : {len(df_fin)}, threshold_ratio : {threshold_ratio}")

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
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)
        # Inference: Generation of the output
        # 2. 모델 추론 & 어텐션 추출
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                return_dict_in_generate=True,
                output_attentions=True, 
            )
        print(f"outputs :{outputs.sequences.shape}")
        input_len = inputs.input_ids.shape[1]
        generated_ids_trimmed = outputs.sequences[0, input_len:]
        output_text = processor.decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        result = output_text
        print(f"{dot} : {result}")
        llM_result_json = parse_llm_json(result)
        dot_res_list.append(llM_result_json['result'])
        dot_reason_list.append(llM_result_json['reason'])
        if llM_result_json['result']=='Pass':
            dot_real_list.append(dot)

        # 3. 어텐션 데이터 준비
        last_step_attentions = outputs.attentions[-1] 
        # print(f"last_step_attentions : {np.shape(last_step_attentions)} ")
        grid_t, grid_h, grid_w = inputs.image_grid_thw[0].cpu().numpy()

        # ✨ 핵심 수정: Qwen2.5-VL은 2x2 풀링을 하므로 그리드 크기를 2로 나눕니다.
        llm_grid_h = grid_h // 2
        llm_grid_w = grid_w // 2
        num_image_tokens = llm_grid_h * llm_grid_w

        print(f"grid_h : {grid_h}, grid_w : {grid_w}, grid_t :{grid_t}, llm_grid_h : {llm_grid_h}, llm_grid_w : {llm_grid_w}")
        
        input_ids = inputs.input_ids[0].cpu().tolist()

        current_img_size = 1000 
        
        target_x = int(dot[0] / current_img_size * llm_grid_w)
        target_y = int(dot[1] / current_img_size *  llm_grid_h)

        # 인덱스 범위 벗어나지 않게 클리핑
        target_x = min(max(target_x, 0), llm_grid_w - 1)
        target_y = min(max(target_y, 0), llm_grid_h - 1)

        valid_heads_count = 0
        accumulated_heatmap = np.zeros((llm_grid_h, llm_grid_w), dtype=np.float32)
        
        print(f"Original Dot: {dot} -> Grid Coords: ({target_x}, {target_y}) / Grid Size: ({llm_grid_w}, {llm_grid_h})")
        all_heads_scores = []
        # 4. 모든 레이어 & 헤드 순회
        for layer_idx, layer_attn in enumerate(last_step_attentions):
            heads_attn = layer_attn[0, :, -1, :] 
            num_heads = heads_attn.shape[0]
            
            for head_idx in range(num_heads):
                # print(f"Layer {layer_idx}, Head {head_idx}")
                this_head_attn = heads_attn[head_idx] 
                # img_attn_1d = this_head_attn[-num_image_tokens:]
                # [수정] 특수 토큰 ID 찾기
                vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
                vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
                
                # input_ids에서 위치 찾기
                # (배치 0번 기준)
                ids_list = inputs.input_ids[0].tolist()
                
                vis_start_idx = ids_list.index(vision_start_id)
                vis_end_idx = ids_list.index(vision_end_id)
                # print (f"vis_start_idx : {vis_start_idx}, vis_end_idx : {vis_end_idx}, vis_end_idx - vis_start_idx : {vis_end_idx - vis_start_idx}, num_image_tokens : {num_image_tokens}")


                # ✨ 핵심: Start 다음부터 ~ End 전까지 (순수 이미지 토큰만)
                # 실제 이미지 토큰 시작 = vis_start_idx + 1
                # 실제 이미지 토큰 끝 = vis_end_idx
                
                # 슬라이싱
                img_attn_1d = this_head_attn[vis_start_idx + 1 : vis_end_idx]
                heatmap_2d = img_attn_1d.reshape(llm_grid_h, llm_grid_w).float().cpu().numpy()
                # 정규화
                if heatmap_2d.max() > 0:
                    heatmap_2d = (heatmap_2d - heatmap_2d.min()) / (heatmap_2d.max() - heatmap_2d.min())
                else:
                    print("heatmap_2d max is 0")
                    continue

                ## 추가부분아래
                point_value = heatmap_2d[target_y, target_x]
                
                # 리스트에 정보 저장 (점수, 레이어, 헤드, 히트맵)
                all_heads_scores.append({
                    "score": point_value,
                    "layer": layer_idx,
                    "head": head_idx,
                    "heatmap": heatmap_2d
                })

                # # 변환된 그리드 좌표(target_x, target_y)로 값 확인
                # point_value = heatmap_2d[target_x, target_y]
                # threshold_value = 0 # heatmap_2d.max() * threshold_ratio                
                # if point_value > threshold_value:
                #     accumulated_heatmap += heatmap_2d
                #     valid_heads_count += 1
                #     # print(f"Point value: {point_value}, max value: {heatmap_2d.max()}, valid_heads_count: {valid_heads_count}")
                #     print(f" Grid Coords: ({target_x}, {target_y}) / ✅ Found valid head! Layer: {layer_idx}, Head: {head_idx} (Val: {point_value:.4f} / Max: {heatmap_2d.max():.4f}), threshold_value: {threshold_value}")

                #     # --- ✨ 히트맵 저장 코드 시작 ---
                #     # 1. 저장할 폴더 만들기 (없으면 생성)
                #     save_dir = "valid_heads_visualization"
                #     os.makedirs(save_dir, exist_ok=True)
                    
                #     # 2. 파일 이름 생성 (레이어_헤드 번호 포함)
                #     # 예: valid_heads_visualization/layer05_head12.png
                #     save_filename = os.path.join(save_dir, f"{filename}_{dot[0]}_{dot[1]}_layer{layer_idx:02d}_head{head_idx:02d}.png")
                    
                #     # 3. 이미지로 저장 (cmap='jet'으로 컬러 히트맵 적용)
                #     # vmin=0, vmax=1 로 고정하면 모든 헤드의 스케일을 통일해서 볼 수 있습니다.
                #     plt.imsave(save_filename, heatmap_2d, cmap='jet', vmin=0, vmax=1)
                #     print(save_filename)
                #     # --- 히트맵 저장 코드 끝 ---
                # else:
                #     print(f"❌ Invalid head! Layer: {layer_idx}, Head: {head_idx} (Val: {point_value:.4f} / Max: {heatmap_2d.max():.4f})")
        all_heads_scores.sort(key=lambda x: x["score"], reverse=True)
        # 2. 상위 5개 선택
        top_k_heads = all_heads_scores[:5]
        
        print(f"✅ Saving Top-{len(top_k_heads)} attention heads...")

        # 3. 저장 폴더 생성
        save_dir = "top_attention_heads"
        os.makedirs(save_dir, exist_ok=True)
        
        # 4. 이미지 저장
        for rank, item in enumerate(top_k_heads):
            score = item["score"]
            layer = item["layer"]
            head = item["head"]
            heatmap = item["heatmap"]
            
            print(f"Rank {rank+1}: Layer {layer}, Head {head}, Score {score:.4f}")
            save_filename = os.path.join(save_dir, f"{dot[0]}_{dot[1]}_rank{rank+1:02d}_L{layer:02d}_H{head:02d}_score_{score:.4f}.png")
            # --- ✨ 수정: 빨간 네모 그리기 ---
            # 1. 캔버스(Figure) 생성 (프레임 없이 이미지 크기에 딱 맞게 설정하려면 조금 복잡하지만, 여기선 보기 좋게 그립니다)
            fig, ax = plt.subplots(figsize=(5, 5))

            # 2. 히트맵 그리기
            ax.imshow(heatmap, cmap='gray_r', vmin=0, vmax=1)
            
            # 3. 빨간 네모 추가
            # Rectangle((x시작, y시작), 너비, 높이, ...)
            # 픽셀의 중심이 (target_x, target_y)이므로, -0.5를 해서 픽셀 테두리에 맞춥니다.
            rect = patches.Rectangle(
                (target_x - 0.5, target_y - 0.5), # 시작 좌표 (x, y)
                1, 1,                             # 너비, 높이 (1칸)
                linewidth=2,                      # 선 굵기
                edgecolor='red',                  # 선 색상
                facecolor='none'                  # 내부 채우기 없음
            )
            ax.add_patch(rect)
            
            # 4. 축 제거 및 저장
            ax.axis('off')
            # 여백 없이 저장 (bbox_inches='tight', pad_inches=0)
            plt.savefig(save_filename, bbox_inches='tight', pad_inches=0)
            plt.close(fig) # 메모리 해제를 위해 닫기

            real_image = Image.open(file_name_real)
            real_w, real_h = real_image.size

            # 2. 히트맵을 원본 이미지 크기로 업샘플링 (Interpolation: Cubic 추천)
            # heatmap은 현재 작은 그리드 크기(예: 31x31)라고 가정
            upsampled_heatmap = cv2.resize(heatmap, (real_w, real_h), interpolation=cv2.INTER_CUBIC)

            # 3. 파일 이름 생성 (요청하신 포맷)
            # dot[0], dot[1]이 float일 수 있으므로 포맷팅을 깔끔하게 하려면 :.0f 등을 추가할 수 있습니다.
            save_filename2 = os.path.join(
                save_dir, 
                f"upsampled_{dot[0]}_{dot[1]}_rank{rank+1:02d}_L{layer:02d}_H{head:02d}_score_{score:.4f}.png"
            )

            # 4. 오버레이 이미지 생성 및 저장
            # 이미지 크기에 맞는 Figure 생성 (DPI 조절로 해상도 유지 가능)
            fig, ax = plt.subplots(figsize=(10, 10)) 

            # (1) 원본 이미지 깔기
            ax.imshow(real_image)

            # (2) 히트맵 덮기 (alpha=0.5로 반투명하게)
            # cmap='jet': 파랑(낮음) -> 빨강(높음) (어텐션 보기에 가장 좋습니다)
            # cmap='gray_r': 흰색(낮음) -> 검은색(높음) (이전 코드 스타일)
            ax.imshow(upsampled_heatmap, cmap='gray_r', alpha=0.5, vmin=0, vmax=1)

            # (3) 축 제거 및 저장
            ax.axis('off')
            plt.savefig(save_filename2, bbox_inches='tight', pad_inches=0)
            plt.close(fig)


        # break
    result_row.append(dot_res_list)
    reason_row.append(dot_reason_list)
    final_dot_row.append(dot_real_list)
    if cnt_d ==0 : 
        break
    cnt_d += 1
df_fin['veri_result'] = result_row
df_fin['veri_reason'] = reason_row
df_fin['final_dot'] = final_dot_row
print(df_fin)

# df_fin.to_pickle('test_verify_qwen3_2b.pkl')

    