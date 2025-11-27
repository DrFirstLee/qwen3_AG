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
'1' : 'jump$skis',
'2' : 'jump$skateboard',
'3' : 'jump$surfboard',
'4' : 'jump$snowboard',
'5' : 'peel$carrot',
'6' : 'peel$orange',
'7' : 'peel$banana',
'8' : 'peel$apple',
'9' : 'wash$toothbrush',
'10' : 'wash$cup',
'11' : 'wash$orange',
'12' : 'wash$fork',
'13' : 'wash$wine_glass',
'14' : 'wash$bowl',
'15' : 'wash$knife',
'16' : 'sit_on$chair',
'17' : 'sit_on$couch',
'18' : 'sit_on$bed',
'19' : 'sit_on$bench',
'20' : 'sit_on$bicycle',
'21' : 'sit_on$motorcycle',
'22' : 'sit_on$skateboard',
'23' : 'sit_on$surfboard',
'24' : 'drag$suitcase',
'25' : 'type_on$laptop',
'26' : 'type_on$keyboard',
'27' : 'pack$suitcase',
'28' : 'cut$carrot',
'29' : 'cut$orange',
'30' : 'cut$banana',
'31' : 'cut$apple',
'32' : 'ride$bicycle',
'33' : 'ride$motorcycle',
'34' : 'cut_with$scissors',
'35' : 'cut_with$knife',
'36' : 'sip$bottle',
'37' : 'sip$cup',
'38' : 'sip$wine_glass',
'39' : 'catch$soccer_ball',
'40' : 'catch$frisbee',
'41' : 'catch$rugby_ball',
'42' : 'lie_on$couch',
'43' : 'lie_on$bed',
'44' : 'lie_on$bench',
'45' : 'lie_on$surfboard',
'46' : 'open$bottle',
'47' : 'open$refrigerator',
'48' : 'open$oven',
'49' : 'open$book',
'50' : 'open$suitcase',
'51' : 'open$microwave',
'52' : 'text_on$cell_phone',
'53' : 'boxing$punching_bag',
'54' : 'stir$bowl',
'55' : 'hit$baseball_bat',
'56' : 'hit$tennis_racket',
'57' : 'hit$hammer',
'58' : 'hit$axe',
'59' : 'write$pen',
'60' : 'take_photo$cell_phone',
'61' : 'take_photo$camera',
'62' : 'pour$bottle',
'63' : 'pour$cup',
'64' : 'pour$wine_glass',
'65' : 'kick$soccer_ball',
'66' : 'kick$rugby_ball',
'67' : 'kick$punching_bag',
'68' : 'pick_up$skis',
'69' : 'pick_up$suitcase',
'70' : 'carry$skis',
'71' : 'carry$skateboard',
'72' : 'carry$surfboard',
'73' : 'carry$snowboard',
'74' : 'stick$fork',
'75' : 'stick$knife',
'76' : 'look_out$binoculars',
'77' : 'hold$toothbrush',
'78' : 'hold$baseball_bat',
'79' : 'hold$bottle',
'80' : 'hold$cup',
'81' : 'hold$scissors',
'82' : 'hold$skis',
'83' : 'hold$tennis_racket',
'84' : 'hold$book',
'85' : 'hold$frisbee',
'86' : 'hold$golf_clubs',
'87' : 'hold$hammer',
'88' : 'hold$fork',
'89' : 'hold$badminton_racket',
'90' : 'hold$suitcase',
'91' : 'hold$wine_glass',
'92' : 'hold$skateboard',
'93' : 'hold$axe',
'94' : 'hold$surfboard',
'95' : 'hold$snowboard',
'96' : 'hold$bowl',
'97' : 'hold$knife',
'98' : 'drink_with$bottle',
'99' : 'drink_with$cup',
'100' : 'drink_with$wine_glass',
'101' : 'brush_with$toothbrush',
'102' : 'throw$soccer_ball',
'103' : 'throw$javelin',
'104' : 'throw$frisbee',
'105' : 'throw$discus',
'106' : 'throw$rugby_ball',
'107' : 'throw$baseball',
'108' : 'throw$basketball',
'109' : 'beat$drum',
'110' : 'lift$fork',
'111' : 'talk_on$cell_phone',
'112' : 'push$bicycle',
'113' : 'push$motorcycle',
'114' : 'eat$hot_dog',
'115' : 'eat$carrot',
'116' : 'eat$orange',
'117' : 'eat$banana',
'118' : 'eat$broccoli',
'119' : 'eat$apple',
'120' : 'swing$baseball_bat',
'121' : 'swing$tennis_racket',
'122' : 'swing$golf_clubs',
'123' : 'swing$badminton_racket',
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
    
    output_file = "3B_exo_selection_all.json"

    # JSON 파일로 저장
    with open(output_file, "w", encoding="utf-16") as f:
        json.dump(res, f, indent=4, ensure_ascii=False, sort_keys=True)