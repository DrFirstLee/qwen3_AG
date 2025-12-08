def process_image_ego_prompt(action, object_name):
    return f"""
        You are a visual reasoning agent specialized in identifying critical keypoints for physical interactions.

        📷 Context:
        You are given an image of size 1000x1000 pixels showing a '{object_name}' involved in the action '{action}'. Your task is to extract **precise, single-point coordinates** and then provide their corresponding semantic part names with relevance labels in JSON format.

        🔍 Task Requirements:
        - First, output a list of **[x, y]** pixel coordinates (integers from 0 to 999) for several semantically meaningful keypoints on the '{object_name}'
        - Points must lie **strictly within the boundaries** of the '{object_name}'
        """+"""
        - Then, output a list of JSON objects in the format: [{"part": "part_name", "label": "positive"}, ...] where:
            - part_name: descriptive name of the object part in one word (e.g., "rim", "handle", "base")
            - label: "positive" if the point is crucial for the action, "negative" if it's irrelevant or non-functional
            - list maximum 5 part name
        - Avoid clustering: ensure points are spatially distinct and representative of different functional regions (minimum distance between any two points: 100 pixels)
        - Do **not** output bounding boxes, masks, or any textual explanations


        🧠 Reasoning Guidance:
        Think about how a human would interact with this object during the action. Consider grip points, pivot areas, contact zones, or structural landmarks essential for the motion.
        Assign "positive" to points that are directly involved in the interaction (e.g., where hands touch, force is applied).
        Assign "negative" to structurally present but functionally irrelevant points (e.g., decorative parts, non-contact surfaces).
        part_names must not be duplicated.

        ✅ Output Format (Strict):
        [
        [x1, y1],
        [x2, y2],
        [x3, y3]
        ]
        [
        {"part": "part_name_1", "label": "positive"},
        {"part": "part_name_2", "label": "negative"},
        {"part": "part_name_3", "label": "positive"}
        ]

        ❌ Prohibited:
        - Any text, labels, comments, or explanations outside these two blocks
        - Coordinates outside the object (x, y ∈ [0, 999])
        - Multiple points too close together (< 100 pixels apart)
        - Bounding boxes or region descriptions
        - Invalid label values (must be exactly "positive" or "negative")

        🎯 Example (for reference only — do not include in output):
        [
        [300, 500],
        [400, 800],
        [600, 450]
        ]
        [
        {"part": "rim", "label": "positive"},
        {"part": "base", "label": "negative"},
        {"part": "handle", "label": "positive"}
        ]

        Now, analyze the image and output ONLY the coordinates and part-label pairs as specified.
    """

def process_image_exo_prompt(action, object_name):
    return f"""
        You are a visual reasoning agent with expertise in cross-view interaction analysis.

        📷 Context:
        You are given two images, both of size 1000x1000 pixels:

        - An **egocentric view** (first-person perspective) showing a '{object_name}' in context.
        - An **exocentric reference image** (third-person or standard view) demonstrating how the action '{action}' is typically performed on the '{object_name}'.

        🎯 Task:
        Identify **precise, single-point keypoints** in the **egocentric image** that correspond to critical interaction locations on the '{object_name}', guided by the exocentric reference.

        🔍 Reasoning Strategy:
        - Analyze the exocentric image to understand the **functional anatomy** of the object during the action (e.g., grip points, contact zones, pivot areas).
        - Map those functional regions to the corresponding parts in the egocentric image.
        - Select keypoints that are **semantically meaningful** and **spatially distinct**.

        📌 Guidelines:
        - All keypoints must lie **strictly within the '{object_name}'** in the egocentric image (coordinates: x, y ∈ [0, 999]).
        - If multiple instances of '{object_name}' exist, mark keypoints on **each instance**.
        - Output **at least 3 well-separated points** covering different functional regions (e.g., ends, center, handle, base). Minimum distance between any two points: 100 pixels.
        - Avoid placing points too close (< 100 pixels apart) or in irrelevant areas.

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
        If the object is a 'hammer' and the action is 'hitting', the exocentric image may show force applied at the head. In the egocentric image, you would select: head center, handle grip, and hammer tip — all as pixel coordinates within the 1000x1000 frame.

        Now, analyze both images and output ONLY the coordinates as specified.
    """