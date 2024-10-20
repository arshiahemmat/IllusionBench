# Simple scenes for the benchmark.
SIMPLE_SCENES = ["ocean", "origami", "forest", "cloud", "sand_dune"]

# Complex scenes for the benchmark.
COMPLEX_SCENES = [
    "medieval_Village",
    "city",
    "underwater_ruins",
    "museum",
    "bazaar_market",
    "time_square",
]

# Combine simple and complex scenes into a single list.
SCENES = SIMPLE_SCENES + COMPLEX_SCENES


# LOGOS dataset tasks split into various categories based on ICL context.
# The final word after the underscore indicates whether the image context includes the logo, the scene, both, or neither.
LOGOS_TC_TASKS = [
    "logos_tc_icl_neither",
    "logos_tc_icl_object",
    "logos_tc_icl_scene",
    "logos_tc_icl_both",
]

LOGOS_TS_TASKS = [
    "logos_ts_icl_neither",
    "logos_ts_icl_object",
    "logos_ts_icl_scene",
    "logos_ts_icl_both",
]

LOGOS_TCS_TASKS = [
    "logos_tcs_icl_neither",
    "logos_tcs_icl_object",
    "logos_tcs_icl_scene",
    "logos_tcs_icl_both",
]

# IN dataset tasks split into categories similar to LOGOS.
IN_TC_TASKS = [
    "in_tc_icl_neither",
    "in_tc_icl_object",
    "in_tc_icl_scene",
    "in_tc_icl_both",
]

IN_TS_TASKS = [
    "in_ts_icl_neither",
    "in_ts_icl_object",
    "in_ts_icl_scene",
    "in_ts_icl_both",
]

IN_TCS_TASKS = [
    "in_tcs_icl_neither",
    "in_tcs_icl_object",
    "in_tcs_icl_scene",
    "in_tcs_icl_both",
]

# ICONS dataset tasks split into categories similar to LOGOS and IN.
ICONS_TC_TASKS = [
    "icons_tc_icl_neither",
    "icons_tc_icl_object",
    "icons_tc_icl_scene",
    "icons_tc_icl_both",
]

ICONS_TS_TASKS = [
    "icons_ts_icl_neither",
    "icons_ts_icl_object",
    "icons_ts_icl_scene",
    "icons_ts_icl_both",
]

ICONS_TCS_TASKS = [
    "icons_tcs_icl_neither",
    "icons_tcs_icl_object",
    "icons_tcs_icl_scene",
    "icons_tcs_icl_both",
]
    

# Combine all tasks for each dataset into separate lists for easy reference.
ALL_LOGOS_TASKS = LOGOS_TC_TASKS + LOGOS_TS_TASKS + LOGOS_TCS_TASKS
ALL_IN_TASKS = IN_TC_TASKS + IN_TS_TASKS + IN_TCS_TASKS
ALL_ICONS_TASKS = ICONS_TC_TASKS + ICONS_TS_TASKS + ICONS_TCS_TASKS

# Combine tasks from all datasets into a single list of all tasks.
ALL_TASKS = ALL_LOGOS_TASKS + ALL_IN_TASKS + ALL_ICONS_TASKS


ICL_BOTH_TASKS = [
    "logos_tc_icl_both",
    "logos_ts_icl_both",
    "logos_tcs_icl_both",
    "in_tc_icl_both",
    "in_ts_icl_both",
    "in_tcs_icl_both",
    "icons_tc_icl_both",
    "icons_ts_icl_both",
    "icons_tcs_icl_both",
]

ICL_SCENE_TASKS = [
    "logos_tc_icl_scene",
    "logos_ts_icl_scene",
    "logos_tcs_icl_scene",
    "in_tc_icl_scene",
    "in_ts_icl_scene",
    "in_tcs_icl_scene",
    "icons_tc_icl_scene",
    "icons_ts_icl_scene",
    "icons_tcs_icl_scene",
]

ICL_SHAPE_TASKS = [
    "logos_tc_icl_object",
    "logos_ts_icl_object",
    "logos_tcs_icl_object",
    "in_tc_icl_object",
    "in_ts_icl_object",
    "in_tcs_icl_object",
    "icons_tc_icl_object",
    "icons_ts_icl_object",
    "icons_tcs_icl_object",
]

ICL_NEITHER_TASKS = [
    "logos_tc_icl_neither",
    "logos_ts_icl_neither",
    "logos_tcs_icl_neither",
    "in_tc_icl_neither",
    "in_ts_icl_neither",
    "in_tcs_icl_neither",
    "icons_tc_icl_neither",
    "icons_ts_icl_neither",
    "icons_tcs_icl_neither",
]
 

# Task dictionary, mapping dataset names to their task lists.
TASKS = {
    "logos": ALL_LOGOS_TASKS,
    "in": ALL_IN_TASKS,
    "icons": ALL_ICONS_TASKS
}


# Logos that all models are expected to recognize during conditioning.
LOGOS = [
    "adidas",
    "mcdonalds",
    "apple",
    "bmw",
    "nike",
    "audi",
    "nasa",
    "spotify",
    "starbucks",
    "tesla",
]

# Object classes within the IN dataset.
IN = [
    "airplane",
    "bicycle",
    "bird",
    "bottle",
    "car",
    "cat",
    "dog",
    "dolphin",
    "fork",
    "guitar",
    "mug",
    "panda",
    "sailboat",
    "scooter",
    "teapot",
]

# Icon classes within the ICONS dataset (representing high-level concepts).
ICONS = ["animal", "face_emoji", "music", "sport", "stationary", "vehicle"]


# Directory paths for datasets used in the tasks (LOGOS, IN, ICONS). ADD THESE IN IF MANUALLY DOWNLOADED THE DATASET.
LOGOS_DATA_FOLDER = ""
IN_DATA_FOLDER = ""
ICON_DATA_FOLDER = ""

# Paths to conditioning images for various datasets (LOGOS, IN, ICONS).
CONDITIONING_IMAGES_LOGOS = (
    ""
)
CONDITIONING_IMAGES_IN = ""
CONDITIONING_IMAGES_ICONS = ""


# List of model engine names to be used in the benchmark tasks.
MODELS = [
    "otter-mpt",
    "llava16-7b",
    "qwen-vl-chat",
    "idefics-9b-instruct",
    "mmicl-t5-xxl",
]

# Number of reruns for the inference process (used for repetition to ensure stability of results).
NUM_RERUNS = 1

# Dataset sizes for various benchmarks (number of examples in each dataset).
DATASET_SIZES = {
    "icons": 20064,
    "logos": 2860,
    "in": 6435
}
