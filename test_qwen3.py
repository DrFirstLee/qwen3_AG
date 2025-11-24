# nohup python -u ego_only.py > GPT5_relative_coord.log 2>&1 & tail -f GPT5_relative_coord.log
import os
import torch
import random
from PIL import Image
# import my_prompt4_gpt as my_prompt
import my_prompt4 as my_prompt
from file_managing import (
    load_selected_samples,
    get_actual_path,
    get_gt_path,
)
from config import AGD20K_PATH, model_name
# from VLM_model_dot_gpt import QwenVLModel, MetricsTracker
from VLM_model_dot import QwenVLModel, MetricsTracker

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ENABLE_SDPA"] = "1"


model = QwenVLModel(model_name = model_name)
res = model.ask("who are you?")
print(res)

res_img = model.ask_with_image("tell me about the image?",f"{AGD20K_PATH}/Seen/trainset/egocentric/boxing/punching_bag/punching_bag_000063.jpg")
print(res_img)