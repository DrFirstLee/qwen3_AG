import os
import torch
import random
from pathlib import Path
import random

from PIL import Image
import my_prompt5_relative2 as my_prompt
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


def affordance_grounding(model, action, object_name, image_path, gt_path, exo_path=None, exo_type=None):
    """
    Process each image using Qwen VL model
    """
    # print(f"Processing image: Action: {action}, Object: {object_name}, Image path: {image_path.split('/')[-1]}, GT path: {gt_path.split('/')[-1]}, Image exists: {os.path.exists(image_path)}, GT exists: {os.path.exists(gt_path)}")
    

    if exo_path is None:
        prompt = my_prompt.process_image_ego_prompt(action, object_name)
               
        results = model.process_image_ego(image_path, prompt, gt_path, action)

        
    else:

        prompt = my_prompt.process_image_exo_prompt(action, object_name)
        results = model.process_image_exo(image_path, prompt, gt_path, exo_path, action, exo_type)

    return results


import json

with open("/root/qwen3_AG/results_32B_absolute/3B_exo_best_image.json", 'r', encoding='utf-8') as f:
    best_exo_json = json.load(f)


def main():
    # Initialize Qwen VL model
    cnt = 0 
    missing_gt = 0
    model = QwenVLModel(model_name = model_name)
    metrics_tracker_ego = MetricsTracker(name="only_ego")
    metrics_tracker_exo_best = MetricsTracker(name="with_exo_best")

    json_path = os.path.join("selected_samples.json")
    data = load_selected_samples(json_path)

    # Get total number of samples
    total_samples = len(data['selected_samples'])
    
    # Process each sample
    print(f"Processing {total_samples} samples...")
    print("=" * 50)    
    for pair_key, sample_info in data["selected_samples"].items():
        print(f"--- Start  {cnt}/{total_samples}", "-"*80) 
        
        action = sample_info["action"]
        object_name = sample_info["object"]
        image_path = get_actual_path(sample_info["image_path"])
        gt_path = get_gt_path(image_path)    
        print(f"Action : {action}, Object : {object_name} image_name : {image_path.split('/')[-1]}")
        exo_best_path = Path(f"{AGD20K_PATH}/Seen/trainset/exocentric/{action}/{object_name}")

        # 이미지 확장자 목록
        valid_ext = {".jpg", ".PNG", ".png", ".JPG"}

        all_images = [p for p in exo_best_path.rglob("*")
                    if p.suffix.lower() in valid_ext]
        if  len(all_images)==0:
            print(f"NO SEEN DATA SET : {action}/{object_name}")
            continue

        # 랜덤 이미지 선택
        random_exo_image = random.choice(all_images)
        # try:
        #     exo_name = best_exo_json[f'{action}${object_name}'] 
        # except:
        #     print(f"ERROR - no exo!! {action}${object_name} ")
        #     continue
        # random_exo_image = Path(f"{AGD20K_PATH}/Seen/trainset/exocentric/{action}/{object_name}/{exo_name}") # 
            
        # with exo random
        results_exo_best = affordance_grounding(model, action, object_name, image_path, gt_path, str(random_exo_image)     )
        metrics_exo_best = results_exo_best['metrics']

        if metrics_exo_best:
            metrics_tracker_exo_best.update(metrics_exo_best)
            metrics_tracker_exo_best.print_metrics(metrics_exo_best, image_path.split('/')[-1])
            
           
        # Count missing GT files
        if not os.path.exists(gt_path):
            missing_gt += 1

        print("*** End  ", "*"*150)
        print("\n\n")
        cnt += 1
    # Print final summary
    print("=" * 50)
    print(f"Total number of action-object pairs processed: {total_samples}")
    print(f"Number of missing GT files: {missing_gt}")
    print(f"All images successfully processed!")

if __name__ == "__main__":
    main() 