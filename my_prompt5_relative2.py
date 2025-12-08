def process_image_ego_prompt(action, object_name):
    return f"""
    You are given an image showing a '{object_name}' involved in the action '{action}'.

    🎯 Your task:
    Identify several **precise keypoints** in the image that are essential for performing the action '{action}' on the object '{object_name}'.

    📏 Coordinate System (CRITICAL):
    - Imagine the image is resized to a **1000x1000** grid.
    - **X-axis**: 0 (left) to 1000 (right)
    - **Y-axis**: 0 (top) to 1000 (bottom)
    - Return coordinates as **integers** within this [0, 1000] range.

    ⚠️ Important Instructions:
    - Only return **single-point** coordinates in the format [x, y]
    - Do **not** return bounding boxes or regions
    - All points must lie **within** the '{object_name}'
    - Avoid placing multiple points too close together
    - ❌ Do **not** include any text, comments, or labels
    - ❌ Do **not** use normalized floats (e.g., 0.5); use integers (e.g., 500)

    ✅ Output format (strict):
    [
      [x1, y1],
      [x2, y2],
      [x3, y3]
    ]
    """

def process_image_exo_prompt(action, object_name):
    return f"""
    You are given two images:
    1. An egocentric image (**Target**) where you must select keypoints.
    2. An exocentric image (**Reference**) showing how the action '{action}' is typically performed on the '{object_name}' with human.

    🎯 Task:
    Select multiple [x, y] keypoints in the **First Image**(egocentric image) that are critical for performing the action '{action}' on the '{object_name}'.

    📏 Coordinate System (CRITICAL):
    - Apply a **1000x1000** grid to the  **First Image**(egocentric image).
    - **X-axis**: 0 (left) to 1000 (right)
    - **Y-axis**: 0 (top) to 1000 (bottom)
    - Return coordinates as **integers** within this [0, 1000] range.

    🔍 Use the **Second Image**(exocentric image) to:
    - Understand typical interaction patterns
    - Identify functionally important parts (e.g., contact or force areas)

    📌 Guidelines:
    - Keypoints must lie **within** the '{object_name}' in the  **First Image**(egocentric image)
    - If there are multiple '{object_name}' instances, mark keypoints on **each of them**
    - Place **at least 3 well-separated** points covering the entire functional region
    - e.g., for a handle: both ends and the center
    - Avoid clustering or irrelevant placements

    ⛔ Do NOT:
    - Include text, labels, bounding boxes, or extra formatting
    - Use normalized floats (e.g., 0.5); strictly use integers (e.g., 500)

    ✅ Output format (strict):
    [
      [x1, y1],
      [x2, y2],
      [x3, y3]
    ]
    """