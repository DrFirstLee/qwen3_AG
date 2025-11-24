import os
import json
from pathlib import Path
from config import AGD20K_PATH


def load_selected_samples(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_actual_path(path_with_variable):
    """Convert ${AGD20K_PATH} to actual path"""
    return path_with_variable.replace("${AGD20K_PATH}", AGD20K_PATH)

def get_gt_path(image_path):
    """
    Convert image path to corresponding GT path
    Example:
    /path/Seen/testset/egocentric/action/object/image.jpg
    -> /path/Seen/testset/GT/action/object/image.png
    """
    parts = image_path.split('/')
    # Find the index of 'testset' in the path
    testset_idx = parts.index('testset')
    # Replace 'egocentric' with 'GT' and change extension to .txt
    parts[testset_idx + 1] = 'GT'
    base_name = os.path.splitext(parts[-1])[0]
    parts[-1] = base_name + '.png'
    return '/'.join(parts)

