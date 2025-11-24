def process_image_ego_prompt(action, object_name):
    return f"""
        You are given an image showing a '{object_name}' involved in the action '{action}'.

        🎯 Task:
        Select multiple **precise keypoints** in the image that are essential for performing the action '{action}' on the '{object_name}'.

        🔍 Guidelines:
        - Focus on areas of **human interaction** or **force application** (e.g., handles, grips, pedals)
        - Cover the **entire functional region**, not just one spot  
        - e.g., for a handle: both ends **and** center
        - If multiple '{object_name}' instances are present, mark keypoints on **each of them**
        - Place **at least 3 well-separated** points **within** the object(s)
        - Avoid clustered or irrelevant points

        ⛔ Do NOT:
        - Include any text, labels, or bounding boxes

        ✅ Output format (strict):
        [
        [x1, y1],
        [x2, y2],
        [x3, y3]
        ]
        """

def ask_image_ego_prompt(action, object_name):
    return f"""
        You are given an image showing a '{object_name}' involved in the action '{action}'.

        🎯 Task:
        Describe precise regions (in words) that are essential for performing '{action}' on the '{object_name}'.

        🔍 Guidelines:
        - Focus on human-interaction or force-application areas (e.g., handles, grips, pedals, joints, switches, edges used to push/pull).
        - Cover the entire functional region (e.g., both ends and the center of a handle).
        - If multiple '{object_name}' instances are present, describe keypoints on each instance.
        - Be concrete and spatially specific (e.g., "center of the handle", "left hinge", "front-right edge").
        - Start each sentence with the part/region name and, if helpful, mention how it supports '{action}'.
        - Do NOT mention pixel coordinates, sizes, bounding boxes, labels, or add explanations beyond the sentences.

        ✅ Output format (strict):
        - If there is one '{object_name}' in the image,
        [The entire {object_name} area in the center of the image]
        - If there are multiple '{object_name}' in the image,
        [[The center of the {object_name} on the left side of the image], [The tip and butt of the {object_name} on the right side of the image]]
        """

def process_image_exo_prompt(action, object_name):
    return f"""
    You are given two images:
    1. An **egocentric** image where you must select keypoints.
    2. An **exocentric** reference image showing how the action '{action}' is typically performed on the '{object_name}'.

    🎯 Task:
    Select multiple [x, y] keypoints in the **egocentric image** that are critical for performing the action '{action}' on the '{object_name}'.

    🔍 Use the exocentric image to:
    - Understand typical interaction patterns
    - Identify functionally important parts (e.g., contact or force areas)

    📌 Guidelines:
    - Keypoints must lie **within** the '{object_name}' in the egocentric image
    - If there are multiple '{object_name}' instances, mark keypoints on **each of them**
    - Place **at least 3 well-separated** points covering the entire functional region
    - e.g., for a handle: both ends and the center
    - Avoid clustering or irrelevant placements

    ⛔ Do NOT:
    - Include text, labels, bounding boxes, or extra formatting

    ✅ Output format (strict):
    [
    [x1, y1],
    [x2, y2],
    [x3, y3]
    ]
    """
