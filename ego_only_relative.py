# nohup python -u ego_only.py > GPT5_relative_coord.log 2>&1 & tail -f GPT5_relative_coord.log
import os
import torch
import random
import json
from PIL import Image
import my_prompt5_relative_clipprompt as my_prompt # my_prompt_qwen3_clipprompt 
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




def affordance_grounding(model, action, object_name, image_path, gt_path, exo_path=None,  failed_heatmap_path=None, validation_reason=None):
    """
    Process each image using Qwen VL model
    """
    print(f"Processing image: Action: {action}, Object: {object_name}, Image path: {image_path}, GT path: {gt_path}, Image exists: {os.path.exists(image_path)}, GT exists: {os.path.exists(gt_path)}")
    

    if exo_path is None:
        prompt = my_prompt.process_image_ego_prompt(action, object_name)
               
        results = model.process_image_ego(image_path, prompt, gt_path, action)

    else:

        prompt = my_prompt.process_image_exo_prompt(action, object_name)
        results = model.process_image_exo(image_path, prompt, gt_path, exo_path, action)

    return results


def main():
    # Initialize Qwen VL model
    model = QwenVLModel(model_name = model_name)
    metrics_tracker_ego = MetricsTracker(name="only_ego")

    json_path = os.path.join("selected_samples.json")
    data = load_selected_samples(json_path)
    missing_gt = 0
    processed_count = 0
    part_info = {}

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
        image_name = image_path.split('/')[-1]
        print(f"Action : {action}, Object : {object_name} image_name : {image_name}")
        # Process the image
        results_ego = affordance_grounding(model, action, object_name, image_path, gt_path)
        metrics_ego = results_ego['metrics']
        parts_info = results_ego['parsed_part']
        part_info[f"{action}${object_name}${image_name}"] = parts_info
        if metrics_ego:
            # Update and print metrics
            metrics_tracker_ego.update(metrics_ego)
            metrics_tracker_ego.print_metrics(metrics_ego, image_path.split('/')[-1])
                    
        # Count missing GT files
        if not os.path.exists(gt_path):
            missing_gt += 1
        with open("part_info_result_3B.json", "w") as f:
            json.dump(part_info, f, indent=4)
        print("*** End  ", "*"*150)
        print("\n\n")
        # break

    # Print final summary
    print("=" * 50)
    print(f"Total number of action-object pairs processed: {total_samples}")
    print(f"Number of missing GT files: {missing_gt}")
    print(f"All images successfully processed!")

if __name__ == "__main__":
    main() 