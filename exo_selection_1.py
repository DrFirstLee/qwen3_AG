import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
from config import AGD20K_PATH, model_name

import glob
import os
import json

# 1. 모델 및 프로세서 로드 (사용자님 Qwen3 코드 기반)
print(f"🤖 {model_name} 모델 로딩중...")

# Qwen3 클래스 사용
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name, dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_name)
tokenizer = processor.tokenizer

print("✅ 모델 로딩 완료!")

# 36개 행동 리스트 (예시로 몇 개만 작성, 실제로는 36개 다 채우시면 됩니다)
actions =  [
    "beat", "brush_with", "catch", "cut_with", "drink_with", "hit", "jump", "lie_on", "look_out", "pack",
    "pick_up", "push", "sip", "stick", "swing", "talk_on", "throw", "wash",
    "boxing", "carry", "cut", "drag", "eat", "hold", "kick", "lift", "open", "peel", "pour", "ride",
    "sit_on", "stir", "take_photo", "text_on", "type_on", "write"
]

patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]

action_token_map = {}
for action in actions:
    # add_special_tokens=False로 순수 단어의 ID만 가져옵니다.
    ids = tokenizer.encode(action, add_special_tokens=False)
    if ids:
        # 첫 번째 토큰 ID만 사용 (대부분의 단일 단어 동사는 1개 토큰입니다)
        action_token_map[action] = ids[0]

print(f"Action Token Map: {action_token_map}")
target_item_dict = {
    "1":"push$bicycle",
    "2" : "hold$badminton_racket",
    "3" : "hold$axe"
}


res = {}
for num in target_item_dict.keys():
    action, object_name = target_item_dict[num].split('$')
    print(f"action, object_name : {action}, {object_name}")
    image_dir = f"{AGD20K_PATH}/Seen/trainset/exocentric/{action}/{object_name}"

    # png, jpg 등 여러 확장자를 모두 포함하고 싶으면:
    
    image_paths = []
    for p in patterns:
        image_paths.extend(glob.glob(os.path.join(image_dir, "**", p), recursive=True))
    image_res = {}
    for image_path in image_paths:
        print(f"file name : {image_path}")
        # 3. 입력 구성
        # 질문: "사람이 [객체]와 무엇을 하고 있는가?" (단답형 유도)
        query = f"What is the person doing with the {object_name}? Answer with a single verb."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": query},
                ],
            }
        ]

        # 4. 입력 전처리
        # apply_chat_template으로 텍스트 프롬프트 생성
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 이미지 로드 및 입력 텐서 생성
        image = Image.open(image_path).convert("RGB")
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        # 5. 모델 추론 (Forward Pass)
        # generate() 대신 model()을 호출하여 raw logits를 얻습니다.
        with torch.no_grad():
            outputs = model(**inputs)
            
            # outputs.logits shape: [batch_size, seq_len, vocab_size]
            # 우리는 '다음 토큰'을 예측하고 싶으므로 시퀀스의 가장 마지막(-1) 로짓을 가져옵니다.
            next_token_logits = outputs.logits[0, -1, :]

        # 6. 로짓 프로빙 (Logit Probing) & 랭킹
        action_scores = {}
        for action, token_id in action_token_map.items():
            # 해당 행동 토큰의 점수(logit)만 쏙 뽑아서 저장
            score = next_token_logits[token_id].item()
            action_scores[action] = score

        # 점수가 높은 순서대로 정렬
        sorted_actions = sorted(action_scores.items(), key=lambda x: x[1], reverse=True)
        best_action = sorted_actions[0][0]

        # --- 결과 출력 ---
        # print(f"\n🖼️ Image: {image_path}")
        # print(f"❓ Query: {query}")
        # print(f"\n📊 One-Token Logit Ranking Result:")
        # for rank, (act, score) in enumerate(sorted_actions, 1):
        #     print(f"{rank}. {act}: {score:.4f}")
        image_res[os.path.basename(image_path)] = action_scores
    res[target_item_dict[num]] = image_res
    
    output_file = "results_vlm_exo_selection.json"

    # JSON 파일로 저장
    with open(output_file, "w", encoding="utf-16") as f:
        json.dump(res, f, indent=4, ensure_ascii=False, sort_keys=True)