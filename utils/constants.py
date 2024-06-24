# Define the tasks within the benchmark that are tradional classification tasks.
CLASSIFICATION_TASKS = ["matching_mi"]

# Define the tokens that are used for the classification tasks.
MATCHING_MI_TOKENS = ["No", "Yes"]

# Indices for the classification tokens.
MATCHING_MI_INDEX = {"No": 0, "Yes": 1}

# Dictionary of classification token indices based on the dataset.
CLASSIFICATION_TOKEN_INDICES = {"matching_mi": MATCHING_MI_INDEX}

SIMPLE_DOMAINS = ["ocean", "origami", "forest", "cloud", "sand_dune"]
COMPLEX_DOMAINS = [
    "medieval_Village",
    "city",
    "underwater_ruins",
    "museum",
    "bazaar_market",
    "time_square",
]
DOMAINS = SIMPLE_DOMAINS + COMPLEX_DOMAINS

MESSAGES_DOMAINS = ["city", "forest", "island", "village", "bazaar_market"]

LOGOS_TASKS = [
    "logos_neither",
    "logos_logo",
    "logos_background",
    "logos_both",
]
LOGOS_BACKGROUND_TASKS = [
    "logos_background_neither",
    "logos_background_logo",
    "logos_background_background",
    "logos_background_both",
]

LOGOS_BOTH_TASKS = [
    "logos_both_neither",
    "logos_both_logo",
    "logos_both_background",
    "logos_both_both",
]

SIN_TASKS = [
    "sin_neither",
    "sin_object",
    "sin_background",
    "sin_both",
]

SIN_BACKGROUND_TASKS = [
    "sin_background_neither",
    "sin_background_object",
    "sin_background_background",
    "sin_background_both",
]

SIN_BOTH_TASKS = [
    "sin_both_neither",
    "sin_both_object",
    "sin_both_background",
    "sin_both_both",
]


ICONS_TASKS = [
    "icons_neither",
    "icons_icon",
    "icons_background",
    "icons_both",
]

ICONS_BACKGROUND_TASKS = [
    "icons_background_neither",
    "icons_background_icon",
    "icons_background_background",
    "icons_background_both",
]

ICONS_BOTH_TASKS = [
    "icons_both_neither",
    "icons_both_icon",
    "icons_both_background",
    "icons_both_both",
]


ALL_LOGOS_TASKS = LOGOS_TASKS + LOGOS_BACKGROUND_TASKS + LOGOS_BOTH_TASKS
ALL_SIN_TASKS = SIN_TASKS + SIN_BACKGROUND_TASKS + SIN_BOTH_TASKS
ALL_ICONS_TASKS = ICONS_TASKS + ICONS_BACKGROUND_TASKS + ICONS_BOTH_TASKS
ALL_TASKS = ALL_LOGOS_TASKS + ALL_SIN_TASKS + ALL_ICONS_TASKS


# LOGOS = [
#     "adidas",
#     "amazon",
#     "benz",
#     "facebook",
#     "google",
#     "ibm",
#     "linkedin",
#     "mcdonalds",
#     "apple",
#     "bmw",
#     "nike",
#     "audi",
#     "instagram",
#     "playstation",
#     "puma",
#     "nasa",
#     "olympics",
#     "pepsi",
#     "spotify",
#     "starbucks",
#     "reebok",
#     "telegram",
#     "tesla",
#     "ubuntu",
# ]

# Logos all models recognise (that they recognise the conditioning image)
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

# SIN = [
#     "airplane",
#     "bicycle",
#     "bird",
#     "bottle",
#     "car",
#     "cat",
#     "dog",
#     "dolphin",
#     "fork",
#     "guitar",
#     "mug",
#     "panda",
#     "paper_clip",
#     "sailboat",
#     "scooter",
#     "teapot",
# ]

SIN = [
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

ICONS = ["animal", "face_emoji", "music", "sport", "stationary", "vehicle"]


LOGOS_DATA_FOLDER = "/homes/55/arshia/illusion-diffusion/illusion_generation/logo/"

SIN_DATA_FOLDER = "/homes/55/arshia/illusion-diffusion/illusion_generation/sin/"

ICON_DATA_FOLDER = "/homes/55/arshia/illusion-diffusion/illusion_generation/icon/"

MESSAGES_DATA_FOLDER = (
    "/homes/55/arshia/illusion-diffusion/illusion_generation/secret_message/other/"
)

CONDITIONING_IMAGES_LOGOS = (
    "/homes/55/arshia/illusion-diffusion/Logo_dataset/ICON/ICON/"
)

CONDITIONING_IMAGES_SIN = "/homes/55/arshia/illusion-diffusion/Sin/ICON/"

CONDITIONING_IMAGES_ICONS = "/homes/55/arshia/illusion-diffusion/Icon_dataset"


MODELS = [
    "otter-mpt",
    "llava16-7b",
    "qwen-vl-chat",
    "idefics-9b-instruct",
    "mmicl-t5-xxl",
]

NUM_RERUNS = 1

DATASET_SIZES = {
    "icons": 20064,
    "logos": 2860,
    "sin": 6435
}