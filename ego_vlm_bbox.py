# nohup python -u ego_only.py > GPT5_relative_coord.log 2>&1 & tail -f GPT5_relative_coord.log
import os
import json
import torch
import random
from PIL import Image
import my_prompt4 as my_prompt
from file_managing import (
    load_selected_samples,
    get_actual_path,
    get_gt_path,
)
from config import AGD20K_PATH, model_name
from VLM_model_dot import QwenVLModel, MetricsTracker

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ENABLE_SDPA"] = "1"

res_text = {}

def main():
    # Initialize Qwen VL model
    model = QwenVLModel(model_name = model_name)
    metrics_tracker_ego = MetricsTracker(name="only_ego")

    json_path = os.path.join("selected_samples.json")
    data = load_selected_samples(json_path)
    missing_gt = 0
    processed_count = 0

    # Get total number of samples
    total_samples = len(data['selected_samples'])
    
    # Process each sample
    print(f"Processing {total_samples} samples...")
    print("=" * 50)    
    for pair_key, sample_info in data["selected_samples"].items():
        processed_count += 1
        print(f"--- Start  {processed_count}  / {total_samples}", "-"*80) 
        
        action = sample_info["action"]
        object_name = sample_info["object"]

        image_path = get_actual_path(sample_info["image_path"])
        gt_path = get_gt_path(image_path)   
        print(f"Action : {action}, Object : {object_name} image_name : {image_path.split('/')[-1]}")
        # Process the image
        raw_prompt = f"""You are an expert in affordance detection.
         which part of {object_name} do people use for action '{action}'? 
         You must output exactly one bounding box for that part.

        Bounding box format:
        - Use pixel coordinates in the form [x_min, y_min, x_max, y_max].
        - All values must be integers.
        - Coordinate origin (0, 0) is at the top-left corner of the image.

        """
        res_from_img = model.ask_with_image(raw_prompt, image_path)
        print(res_from_img)
        res_text[f"{object_name}${action}${image_path.split('/')[-1]}"] = res_from_img

        output_file = "results/results_vlm_bbox.json"

        # JSON 파일로 저장
        with open(output_file, "w", encoding="utf-16") as f:
            json.dump(res_text, f, indent=4, ensure_ascii=False, sort_keys=True)


    # Print final summary
    print("=" * 50)
    print(f"All images successfully processed!")

if __name__ == "__main__":
    main() 