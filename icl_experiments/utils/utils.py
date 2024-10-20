import torch
import os
import random
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
from .constants import (
    LOGOS,
    IN,
    ICONS,
    CONDITIONING_IMAGES_IN,
    CONDITIONING_IMAGES_LOGOS,
    CONDITIONING_IMAGES_ICONS,
)



def set_random_seed(seed_number):
    """Set the seed for reproducibility across numpy, random, and torch modules."""
    os.environ["PYTHONHASHSEED"] = str(seed_number)
    np.random.seed(seed_number)
    random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.random.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def truncate_prediction(prediction: str) -> str:
    """Truncate captions at the first newline or period, removing leading spaces."""
    prediction = prediction.strip()
    trunc_index = prediction.find("\n")
    if trunc_index != -1:
        prediction = prediction[:trunc_index].strip()
    else:
        trunc_index = prediction.find(".") + 1
        if trunc_index > 0:
            prediction = prediction[:trunc_index].strip()
    return prediction


def load_image(img_ids, root_path):
    """Load image(s) from file system given image ID(s) and root directory."""
    if isinstance(img_ids, str):
        img_ids = [img_ids]
    images = []
    image_paths = []
    for img_id in img_ids:
        image_path = os.path.join(root_path, img_id)
        image = Image.open(image_path).convert("RGB")
        images.append(image)
        image_paths.append(image_path)
    return images, image_paths


## load data
def load_data(args):
    """Load query and support metadata from json files."""
    dataDir = args.dataDir
    query_file = os.path.join(dataDir, args.dataset, "query.json")
    support_file = os.path.join(dataDir, args.dataset, "support.json")

    with open(query_file, "r") as f:
        query_meta = json.load(f)
    with open(support_file, "r") as f:
        support_meta = json.load(f)

    return query_meta, support_meta



def load(data_dir, dataset):
    """Load the logos or related dataset and organize the data by scene type and logo for OCR."""
    if "logos" in dataset:
        classes = LOGOS
    elif "in" in dataset:
        classes = IN
    elif "icons" in dataset:
        classes = ICONS
    else:
        raise ValueError("Dataset not recognized!")

    shapes = {}  # Dictionary to store images organized by scene and shape (logo, icon, etc.)

    # Iterate over prompt types (simple and complex)
    prompt_types = ["simple_prompts", "complex_prompts"]
    for prompt_type in prompt_types:
        prompt_directory = os.path.join(data_dir, prompt_type)
        for shape in classes:
            for type_label in ["Easy", "Hard"]:
                shape_dir = get_shape_directory(shape, dataset)
                directory = os.path.join(prompt_directory, shape_dir, type_label)

                # Safely attempt to list files in the directory
                try:
                    file_names = os.listdir(directory)
                except FileNotFoundError:
                    print(f"Directory not found: {directory}")
                    continue

                for file_name in file_names:
                    if file_name.endswith(".png"):
                        scene = get_scene_label(file_name, dataset)
                        # Initialize nested dictionary structure
                        shapes.setdefault(scene, {}).setdefault(shape, [])
                        # Append file path
                        shapes[scene][shape].append(os.path.join(directory, file_name))

    return shapes


def get_shape_directory(shape, dataset):
    """Helper to handle shape directory name formatting for different datasets."""
    if "logos" in dataset:
        return shape
    elif "in" in dataset:
        return shape.capitalize()
    elif "icons" in dataset:
        return shape[0].upper() + shape[1:5] + shape[5].upper() + shape[6:] if shape == "face_emoji" else shape.capitalize()
    else:
        raise ValueError("Dataset not recognized!")


def get_scene_label(file_name, dataset):
    """Helper to extract scene label from file names."""
    parts = file_name.split("-")
    if "icons" in dataset:
        return parts[2]
    return parts[1]



def load_logo_conditioning(dataset):
    """Load conditioning images for logos, IN, or icons and organize by shape."""
    if dataset == "logos_conditioning":
        data_dir = CONDITIONING_IMAGES_LOGOS
    elif dataset == "in_conditioning":
        data_dir = CONDITIONING_IMAGES_IN
    elif dataset == "icons_conditioning":
        data_dir = CONDITIONING_IMAGES_ICONS

    shapes = {}

    for folder in os.listdir(data_dir):
        if folder == ".DS_Store":
            continue
        obj = folder.lower()
        shapes[obj] = []
        for file_name in os.listdir(os.path.join(data_dir, folder)):
            if file_name.endswith(".png"):
                shapes[obj].append(os.path.join(data_dir, folder, file_name))

    return shapes


def save_image_pairs(original_images, processed_images, save_path):
    """Helper function to save pairs of original and processed images"""
    for i, (orig_img, proc_img) in enumerate(zip(original_images, processed_images)):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        axs[0].imshow(orig_img)
        axs[0].axis("off")
        axs[0].set_title("Original Image")

        proc_img = (proc_img - proc_img.min()) / (proc_img.max() - proc_img.min())
        axs[1].imshow(proc_img)
        axs[1].axis("off")
        axs[1].set_title("Processed Image")

        # Create the save path if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        plt.savefig(os.path.join(save_path, f"image_pair_{i}.png"))
        plt.close()