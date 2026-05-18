"""
TACTILE — Task definitions for COCO-Tasks (14 Tasks from Sawatzky et al., CVPR 2019).

Each task maps to a set of COCO category IDs that are considered "preferred" objects
for that task. The mapping is derived from the COCO-Tasks annotation statistics.
"""

# ─── 14 TASK DEFINITIONS ───────────────────────────────────────────────────────.
# task_id (0-indexed) -> (task_name, description).
TASK_NAMES = {
    0:  "Serve wine",
    1:  "Spread butter / jam",
    2:  "Drink coffee",
    3:  "Set the table",
    4:  "Cut vegetables",
    5:  "Serve food on a plate",
    6:  "Tighten a screw",
    7:  "Dig a hole",
    8:  "Hang a picture",
    9:  "Check the time",
    10: "Make a phone call",
    11: "Take a photo",
    12: "Play music",
    13: "Read a book",
}

NUM_TASKS = 14

# ─── COCO-Tasks annotation file mapping ────────────────────────────────────────.
# In the COCO-Tasks dataset, Tasks are 1-indexed (task_1, task_2, ..., task_14).
# We use 0-indexed internally.
def get_annotation_filename(task_id: int, split: str = "train") -> str:
    """Return annotation filename for a given task_id (0-indexed) and split."""
    assert 0 <= task_id < NUM_TASKS, f"task_id must be in [0, 13], got {task_id}"
    assert split in ("train", "test"), f"split must be 'train' or 'test', got {split}"
    return f"task_{task_id + 1}_{split}.json"

# ─── COCO 80-class names (same order as COCO category IDs) ─────────────────────.
# Note: COCO has 91 category IDs but only 80 actual classes.
# This list is in the order of contiguous class index 0..79 used by YOLO.
COCO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

NUM_COCO_CLASSES = 80

# ─── COCO category ID <-> contiguous class index mapping ───────────────────────.
# COCO category IDs are not contiguous (1-90 with gaps).
# This mapping converts between COCO cat_id and contiguous index used by YOLO.
COCO_CAT_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]

# cat_id -> contiguous index.
COCO_CATID_TO_IDX = {cat_id: idx for idx, cat_id in enumerate(COCO_CAT_IDS)}
# contiguous index -> cat_id.
COCO_IDX_TO_CATID = {idx: cat_id for idx, cat_id in enumerate(COCO_CAT_IDS)}

# ─── Task-relevant COCO classes (heuristic mapping) ────────────────────────────.
# This provides an initial heuristic for which COCO classes are likely preferred.
# for each task. The actual mapping is learned from COCO-Tasks annotations.
TASK_RELEVANT_CLASSES = {
    0:  ["wine glass", "bottle", "cup"],                    # Serve wine
    1:  ["knife", "spoon", "bowl", "dining table"],         # Spread butter/jam
    2:  ["cup", "dining table"],                            # Drink coffee
    3:  ["fork", "knife", "spoon", "bowl", "cup",
         "wine glass", "dining table", "chair"],            # Set the table
    4:  ["knife", "carrot", "broccoli"],                    # Cut vegetables
    5:  ["bowl", "fork", "spoon", "dining table"],          # Serve food on plate
    6:  ["scissors"],                                       # Tighten a screw
    7:  ["potted plant"],                                   # Dig a hole
    8:  ["tv", "clock", "vase"],                            # Hang a picture
    9:  ["clock", "cell phone"],                            # Check the time
    10: ["cell phone"],                                     # Make a phone call
    11: ["cell phone", "laptop", "tv"],                     # Take a photo
    12: ["remote", "laptop", "tv"],                         # Play music
    13: ["book"],                                           # Read a book
}
