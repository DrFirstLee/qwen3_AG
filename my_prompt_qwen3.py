def process_image_ego_prompt(action, object_name):
    return f"""
        You are a visual reasoning agent specialized in identifying critical keypoints for physical interactions.

        📷 Context:
        You are given an image showing a '{object_name}' involved in the action '{action}'. Your task is to extract **precise, single-point coordinates** that represent key interaction points on the object necessary for performing the action.

        🔍 Task Requirements:
        - Identify several **semantically meaningful keypoints** directly on the '{object_name}'
        - Each point must be a **single coordinate [x, y]** (normalized or pixel-based — use consistent format)
        - Points must lie **strictly within the boundaries** of the '{object_name}'
        - Avoid clustering: ensure points are spatially distinct and representative of different functional regions
        - Do **not** output bounding boxes, masks, or any textual explanations

        🧠 Reasoning Guidance:
        Think about how a human would interact with this object during the action. Consider grip points, pivot areas, contact zones, or structural landmarks essential for the motion.

        ✅ Output Format (Strict):
        [
        [x1, y1],
        [x2, y2],
        [x3, y3]
        ]

        ❌ Prohibited:
        - Any text, labels, comments, or explanations
        - Coordinates outside the object
        - Multiple points too close together (< 10% of object width/height apart)
        - Bounding boxes or region descriptions

        🎯 Example (for reference only — do not include in output):
        If the object is a 'cup' and the action is 'lifting', key points might be the rim center, handle grip, and base center.

        Now, analyze the image and output ONLY the coordinates as specified.
    """

def process_image_exo_prompt(action, object_name):
    return f"""
        You are a visual reasoning agent with expertise in cross-view interaction analysis.

        📷 Context:
        You are given two images:
        - An **egocentric view** (first-person perspective) showing a '{object_name}' in context.
        - An **exocentric reference image** (third-person or standard view) demonstrating how the action '{action}' is typically performed on the '{object_name}'.

        🎯 Task:
        Identify **precise, single-point keypoints** in the **egocentric image** that correspond to critical interaction locations on the '{object_name}', guided by the exocentric reference.

        🔍 Reasoning Strategy:
        - Analyze the exocentric image to understand the **functional anatomy** of the object during the action (e.g., grip points, contact zones, pivot areas).
        - Map those functional regions to the corresponding parts in the egocentric image.
        - Select keypoints that are **semantically meaningful** and **spatially distinct**.

        📌 Guidelines:
        - All keypoints must lie **strictly within the '{object_name}'** in the egocentric image.
        - If multiple instances of '{object_name}' exist, mark keypoints on **each instance**.
        - Output **at least 3 well-separated points** covering different functional regions (e.g., ends, center, handle, base).
        - Avoid placing points too close (< 10% of object size apart) or in irrelevant areas.

        ⛔ Prohibited:
        - Any text, comments, labels, or explanations.
        - Bounding boxes, masks, or region descriptions.
        - Points outside the object or not aligned with the action’s functional needs.

        ✅ Output Format (Strict):
        [
        [x1, y1],
        [x2, y2],
        [x3, y3]
        ]

        🧠 Example (for understanding only — do NOT include in output):
        If the object is a 'hammer' and the action is 'hitting', the exocentric image may show force applied at the head. In the egocentric image, you would select: head center, handle grip, and hammer tip.

        Now, analyze both images and output ONLY the coordinates as specified.
    """