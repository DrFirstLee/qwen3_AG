
import pandas as pd
import ast
import re

# def parse_log_to_df(file_path):
#     data_list = []
    
#     # 현재 처리 중인 샘플의 메타데이터 임시 저장용 (action, object, filename)
#     current_meta = None 
#     is_ego_section = False

#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             line = line.strip()
            
#             # 1. Action 라인 파싱 (새로운 Ego 샘플 시작)
#             # 포맷: Action : jump, Object : skis image_name : skis_002829.jpg
#             if line.startswith("Action :"):
#                 # 정규표현식으로 action, object, filename 추출
#                 match = re.search(r"Action\s*:\s*(.*?),\s*Object\s*:\s*(.*?)\s+image_name\s*:\s*(.*)", line)
#                 if match:
#                     action = match.group(1).strip()
#                     obj = match.group(2).strip()
#                     filename = match.group(3).strip()
                    
#                     current_meta = (action, obj, filename)
#                     is_ego_section = True
#                 continue

#             # 2. Exo 라인 감지 (이 이후의 dots는 무시)
#             if line.startswith("exo file name :"):
#                 is_ego_section = False
#                 continue

#             # 3. Dots 파싱 및 데이터 병합
#             if line.startswith("parsed dots!!! :"):
#                 # Ego 섹션이고, 메타데이터가 확보된 상태일 때만 저장
#                 if is_ego_section and current_meta is not None:
#                     dots_str = line.split(":", 1)[1].strip()
#                     dots = ast.literal_eval(dots_str)
                    
#                     # [action, object, filename, dots] 형태로 추가
#                     data_list.append([current_meta[0], current_meta[1], current_meta[2], dots])


#     # --- DataFrame 생성 및 중복 처리 (요청하신 로직) ---
#     df = pd.DataFrame(data_list)
    
#     if not df.empty:
#         df.columns = ['action', 'object', 'filename', 'dots']
        
#         # action, object, filename 조합이 중복되는 경우 제거
#         df_fin = df.loc[df[['action', 'object', 'filename']].drop_duplicates().index].sort_values(['object','action']).reset_index(drop=True)
#     else:
#         # 데이터가 없을 경우 빈 DF 반환
#         df_fin = pd.DataFrame(columns=['action', 'object', 'filename', 'dots'])

#     print(f">>>>> Total data Length : {len(df_fin)}")
#     return df_fin


# file_path = '/home/bongo/porter_notebook/research/qwen3/ego_exo_prompt5_relative.txt'
# df_fin = parse_log_to_df(file_path)

# df_fin['description'] = "apple"
# df_fin.to_pickle("target_df.pkl")

text = """action : cut, object : apple, filename : apple_000054.jpg, description : "apple"
action : eat, object : apple, filename : apple_001541.jpg, description : "fruit"
action : peel, object : apple, filename : apple_001541.jpg, description : "fruit"
action : hit, object : axe, filename : axe_000961.jpg, description : "axe handle"
action : hold, object : axe, filename : axe_001552.jpg, description : "axe handle"
action : hold, object : badminton_racket, filename : badminton_racket_002255.jpg, description : "badminton racket handle"
action : swing, object : badminton_racket, filename : badminton_racket_003649.jpg, description : "badminton racket handle"
action : cut, object : banana, filename : banana_002623.jpg, description : "banana"
action : eat, object : banana, filename : banana_002458.jpg, description : "banana"
action : peel, object : banana, filename : banana_000480.jpg, description : "banana"
action : throw, object : baseball, filename : baseball_002670.jpg, description : "baseball"
action : hit, object : baseball_bat, filename : baseball_bat_001882.jpg, description : "baseball bat"
action : hold, object : baseball_bat, filename : baseball_bat_002547.jpg, description : "baseball bat handle"
action : swing, object : baseball_bat, filename : baseball_bat_001882.jpg, description : "baseball bat handle"
action : throw, object : basketball, filename : basketball_003534.jpg, description : "basketball"
action : lie_on, object : bed, filename : bed_002880.jpg, description : "bed"
action : sit_on, object : bed, filename : bed_003622.jpg, description : "bed"
action : lie_on, object : bench, filename : bench_003727.jpg, description : "bench seat"
action : sit_on, object : bench, filename : bench_001877.jpg, description : "bench seat"
action : push, object : bicycle, filename : bicycle_002432.jpg, description : "bicycle handlebars"
action : ride, object : bicycle, filename : bicycle_003046.jpg, description : "bicycle handlebars.bicycle pedal.bicycle seat"
action : sit_on, object : bicycle, filename : bicycle_002100.jpg, description : "bicycle seat"
action : look_out, object : binoculars, filename : binoculars_003630.jpg, description : "binoculars"
action : hold, object : book, filename : book_001195.jpg, description : "book page"
action : open, object : book, filename : book_003044.jpg, description : "book page"
action : drink_with, object : bottle, filename : bottle_003259.jpg, description : "bottle cap"
action : hold, object : bottle, filename : bottle_001227.jpg, description : "bottle body"
action : open, object : bottle, filename : bottle_001033.jpg, description : "bottle cap"
action : pour, object : bottle, filename : bottle_002780.jpg, description : "bottle body"
action : hold, object : bowl, filename : bowl_000546.jpg, description : "bowl"
action : stir, object : bowl, filename : bowl_000134.jpg, description : "bowl inside"
action : wash, object : bowl, filename : bowl_002825.jpg, description : "inside bowl"
action : eat, object : broccoli, filename : broccoli_002796.jpg, description : "broccoli"
action : take_photo, object : camera, filename : camera_002534.jpg, description : "camera grip"
action : cut, object : carrot, filename : carrot_001443.jpg, description : "carrot"
action : eat, object : carrot, filename : carrot_001443.jpg, description : "carrot"
action : peel, object : carrot, filename : carrot_003707.jpg, description : "carrot"
action : take_photo, object : cell_phone, filename : cell_phone_000601.jpg, description : "cell phone"
action : talk_on, object : cell_phone, filename : cell_phone_000601.jpg, description : "cell phone screen"
action : text_on, object : cell_phone, filename : cell_phone_003361.jpg, description : "cell phone screen"
action : sit_on, object : chair, filename : chair_002839.jpg, description : "chair seat"
action : lie_on, object : couch, filename : couch_003293.jpg, description : "couch seat"
action : sit_on, object : couch, filename : couch_000779.jpg, description : "couch seat"
action : drink_with, object : cup, filename : cup_000508.jpg, description : "cup rim"
action : hold, object : cup, filename : cup_002518.jpg, description : "cup handle"
action : pour, object : cup, filename : cup_001535.jpg, description : "cup handle"
action : sip, object : cup, filename : cup_001864.jpg, description : "cup rim"
action : wash, object : cup, filename : cup_003621.jpg, description : "cup"
action : throw, object : discus, filename : discus_003558.jpg, description : "discus rim"
action : beat, object : drum, filename : drum_002586.jpg, description : "drum"
action : hold, object : fork, filename : fork_000804.jpg, description : "fork handle"
action : lift, object : fork, filename : fork_001691.jpg, description : "fork handle"
action : stick, object : fork, filename : fork_000095.jpg, description : "fork tines"
action : wash, object : fork, filename : fork_001691.jpg, description : "fork tines"
action : catch, object : frisbee, filename : frisbee_000598.jpg, description : "frisbee rim"
action : hold, object : frisbee, filename : frisbee_001130.jpg, description : "frisbee rim"
action : throw, object : frisbee, filename : frisbee_003249.jpg, description : "frisbee rim"
action : hold, object : golf_clubs, filename : golf_clubs_000045.jpg, description : "golf club handle"
action : swing, object : golf_clubs, filename : golf_clubs_001992.jpg, description : "golf club handle"
action : hit, object : hammer, filename : hammer_001006.jpg, description : "hammer handle"
action : hold, object : hammer, filename : hammer_000215.jpg, description : "hammer handle"
action : eat, object : hot_dog, filename : hot_dog_002166.jpg, description : "hot dog"
action : throw, object : javelin, filename : javelin_001474.jpg, description : "javelin handle"
action : type_on, object : keyboard, filename : keyboard_000439.jpg, description : "keyboard"
action : cut_with, object : knife, filename : knife_001749.jpg, description : "knife blade"
action : hold, object : knife, filename : knife_002682.jpg, description : "knife handle"
action : stick, object : knife, filename : knife_001072.jpg, description : "knife blade"
action : wash, object : knife, filename : knife_002720.jpg, description : "knife blade"
action : type_on, object : laptop, filename : laptop_000585.jpg, description : "laptop keyboard"
action : open, object : microwave, filename : microwave_001049.jpg, description : "microwave door handle"
action : push, object : motorcycle, filename : motorcycle_003541.jpg, description : "motorcycle handlebars.motorcycle seat"
action : ride, object : motorcycle, filename : motorcycle_002198.jpg, description : "motorcycle handlebars.motorcycle seat.motorcycle footrest"
action : sit_on, object : motorcycle, filename : motorcycle_000837.jpg, description : "motorcycle seat"
action : cut, object : orange, filename : orange_001193.jpg, description : "orange"
action : eat, object : orange, filename : orange_001193.jpg, description : "orange"
action : peel, object : orange, filename : orange_001193.jpg, description : "orange"
action : wash, object : orange, filename : orange_001193.jpg, description : "orange"
action : open, object : oven, filename : oven_001370.jpg, description : "oven door handle"
action : write, object : pen, filename : pen_003590.jpg, description : "pen grip"
action : boxing, object : punching_bag, filename : punching_bag_001845.jpg, description : "punching bag"
action : kick, object : punching_bag, filename : punching_bag_001639.jpg, description : "punching bag"
action : open, object : refrigerator, filename : refrigerator_002162.jpg, description : "refrigerator door handle"
action : catch, object : rugby_ball, filename : rugby_ball_003522.jpg, description : "rugby ball"
action : kick, object : rugby_ball, filename : rugby_ball_002080.jpg, description : "rugby ball"
action : throw, object : rugby_ball, filename : rugby_ball_000001.jpg, description : "rugby ball"
action : cut_with, object : scissors, filename : scissors_002479.jpg, description : "scissors blade"
action : hold, object : scissors, filename : scissors_002479.jpg, description : "scissors handle"
action : carry, object : skateboard, filename : skateboard_002668.jpg, description : "center of skateboard"
action : hold, object : skateboard, filename : skateboard_002387.jpg, description : "edge of skateboard"
action : jump, object : skateboard, filename : skateboard_002387.jpg, description : "skateboard foot placement area"
action : sit_on, object : skateboard, filename : skateboard_001460.jpg, description : "skateboard top"
action : carry, object : skis, filename : skis_002829.jpg, description : "center of skis"
action : hold, object : skis, filename : skis_001357.jpg, description : "center of skis"
action : jump, object : skis, filename : skis_002829.jpg, description : "skis standing area"
action : pick_up, object : skis, filename : skis_001547.jpg, description : "center of skis"
action : carry, object : snowboard, filename : snowboard_001325.jpg, description : "center of snowboard"
action : hold, object : snowboard, filename : snowboard_001704.jpg, description : "center of snowboard"
action : jump, object : snowboard, filename : snowboard_001704.jpg, description : "snowboard standing area"
action : catch, object : soccer_ball, filename : soccer_ball_003333.jpg, description : "soccer ball"
action : kick, object : soccer_ball, filename : soccer_ball_001588.jpg, description : "soccer ball"
action : drag, object : suitcase, filename : suitcase_002998.jpg, description : "suitcase handle"
action : hold, object : suitcase, filename : suitcase_003687.jpg, description : "suitcase handle"
action : open, object : suitcase, filename : suitcase_000520.jpg, description : "suitcase rim"
action : pack, object : suitcase, filename : suitcase_002212.jpg, description : "inside suitcase"
action : pick_up, object : suitcase, filename : suitcase_002493.jpg, description : "suitcase"
action : carry, object : surfboard, filename : surfboard_002422.jpg, description : "center of surfboard"
action : hold, object : surfboard, filename : surfboard_002631.jpg, description : "surfboard"
action : jump, object : surfboard, filename : surfboard_000658.jpg, description : "surfboard foot placement area"
action : lie_on, object : surfboard, filename : surfboard_000221.jpg, description : "surfboard"
action : sit_on, object : surfboard, filename : surfboard_000010.jpg, description : "surfboard"
action : hit, object : tennis_racket, filename : tennis_racket_002268.jpg, description : "tennis racket handle"
action : hold, object : tennis_racket, filename : tennis_racket_001785.jpg, description : "tennis racket handle"
action : swing, object : tennis_racket, filename : tennis_racket_003066.jpg, description : "tennis racket handle"
action : brush_with, object : toothbrush, filename : toothbrush_001764.jpg, description : "toothbrush bristle"
action : hold, object : toothbrush, filename : toothbrush_003341.jpg, description : "toothbrush handle"
action : wash, object : toothbrush, filename : toothbrush_001991.jpg, description : "toothbrush bristle
action : drink_with, object : wine_glass, filename : wine_glass_003343.jpg, description : "wineglass rim"
action : hold, object : wine_glass, filename : wine_glass_002374.jpg, description : "wineglass neck"
action : pour, object : wine_glass, filename : wine_glass_000186.jpg, description : "wine glass neck"
action : sip, object : wine_glass, filename : wine_glass_003343.jpg, description : "wine glass rim"
action : wash, object : wine_glass, filename : wine_glass_000186.jpg, description : "wine glass body"
"""



# 1. 텍스트 데이터를 줄 단위로 분리
lines = text.strip().split('\n')

data_list = []
for line in lines:
    # 각 줄을 ', ' 기준으로 분할
    parts = line.split(', ')
    row = {}
    for part in parts:
        # ' : ' 기준으로 key와 value 분리
        key_val = part.split(' : ', 1)
        if len(key_val) == 2:
            key = key_val[0].strip()
            # value에서 앞뒤 공백 및 따옴표 제거
            val = key_val[1].strip().strip('"')
            row[key] = val
    data_list.append(row)

# 2. DataFrame 생성
df_fin = pd.DataFrame(data_list)

# 3. 결과 확인
print(len(df_fin))

df_fin.to_pickle("target_df_w_description.pkl")
# df_fin.head() # 데이터 확인용