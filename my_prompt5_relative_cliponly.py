def process_image_ego_prompt(action, object_name):
    return f"""
    You are given an image showing a '{object_name}' involved in the action '{action}'.

    🎯 Your task:
    1. For each point, determine:
       - The **name of the part** (e.g., handle, blade, rim).
       - A **label**: 'positive' if this part is essential/correct for the action '{action}', or 'negative' if it is a part to avoid or irrelevant.


    ✅ Output Format (Strict):
    [
      {{"part": "part_name_1", "label": "positive"}},
      {{"part": "part_name_2", "label": "negative"}},
      {{"part": "part_name_3", "label": "positive"}}
    ]
    """

def process_image_exo_prompt(action, object_name):
    return f"""
    You are given two images:
    1. An **egocentric** image (Target) where you must select keypoints.
    2. An **exocentric** reference image showing how the action '{action}' is typically performed on the '{object_name}'.

    🎯 Task:
    1. Select multiple [x, y] keypoints in the **egocentric image** on the '{object_name}'.
    2. For each point, based on the **exocentric reference**, determine:
       - The **name of the part** (e.g., handle, rim, button).
       - A **label**: 'positive' if the reference image shows this part is critical for the action, or 'negative' if it is irrelevant/unused.

    📏 Coordinate System (CRITICAL):
    - Apply a **1000x1000** grid to the **Egocentric image**.
    - **X-axis**: 0 (left) to 1000 (right)
    - **Y-axis**: 0 (top) to 1000 (bottom)
    - Return coordinates as **integers** within this [0, 1000] range.

    🔍 Use the exocentric image to:
    - Understand typical interaction patterns (how humans hold or touch the object).
    - Distinguish between 'positive' parts (contact areas) and 'negative' parts.

    ⚠️ Important Instructions:
    - Return **two separate lists** strictly following the format below.
    - The first list contains the coordinates [x, y].
    - The second list contains the part info and label.
    - **The order must match exactly.** (The 1st item in the second list describes the 1st coordinate, and so on).
    - ❌ Do **not** use normalized floats (e.g., 0.5); use integers (e.g., 500).

    ✅ Output Format (Strict):
    [
      [x1, y1],
      [x2, y2],
      [x3, y3]
    ]
    [
      {{ "part": "part_name_1", "label": "positive" }},
      {{ "part": "part_name_2", "label": "negative" }},
      {{ "part": "part_name_3", "label": "positive" }}
    ]
    """