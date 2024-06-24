import torch
import os
import random
import numpy as np
import json
import base64
from PIL import Image
import pandas as pd

# import matplotlib.pyplot as plt
from .constants import (
    LOGOS_DATA_FOLDER,
    SIN_DATA_FOLDER,
    LOGOS,
    SIN,
    ICONS,
    SIMPLE_DOMAINS,
    COMPLEX_DOMAINS,
    CONDITIONING_IMAGES_SIN,
    CONDITIONING_IMAGES_LOGOS,
    CONDITIONING_IMAGES_ICONS,
    MESSAGES,
)


try:
    import fitz  # PyMuPDF
except:
    pass
import io


def set_random_seed(seed_number):
    # position of setting seeds also matters
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
    """Truncate captions at the first newline character, removing leading spaces."""
    prediction = prediction.strip()  # Remove leading and trailing whitespace
    trunc_index = prediction.find("\n")
    if trunc_index != -1:
        prediction = prediction[:trunc_index].strip()
    else:
        # If no newline is found, find the first period and truncate
        trunc_index = prediction.find(".") + 1
        if trunc_index > 0:
            prediction = prediction[:trunc_index].strip()
    return prediction


def load_image(img_ids, root_path):
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


def coco_id_to_imgname(img_id, prefix="COCO_val2014_"):
    return f"{prefix}{img_id:012}.jpg"


## load data
def load_data(args):
    dataDir = args.dataDir
    query_file = os.path.join(dataDir, args.dataset, "query.json")
    support_file = os.path.join(dataDir, args.dataset, "support.json")

    with open(query_file, "r") as f:
        query_meta = json.load(f)
    with open(support_file, "r") as f:
        support_meta = json.load(f)

    return query_meta, support_meta


def load_text_data(args):
    dataset = args.dataset
    dataDir = args.dataDir
    if dataset == "agnews":
        from datasets import load_dataset

        data = load_dataset("ag_news")
        support, query = data["train"], data["test"]
        label_dict = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
        support_meta = []
        for s in support:
            support_meta.append(
                {"question": s["text"], "answer": label_dict[s["label"]]}
            )
        query_meta = []
        for q in query:
            query_meta.append({"question": q["text"], "answer": label_dict[q["label"]]})
    elif dataset == "imdb":
        from datasets import load_dataset

        data = load_dataset("imdb")
        support, query = data["train"], data["test"]
        label_dict = {0: "Negative", 1: "Positive"}
        support_meta = []
        for s in support:
            support_meta.append(
                {"question": s["text"], "answer": label_dict[s["label"]]}
            )
        query_meta = []
        for q in query:
            query_meta.append({"question": q["text"], "answer": label_dict[q["label"]]})
        query_meta = query_meta[:1000]
    elif dataset == "trec":
        from datasets import load_dataset

        data = load_dataset("trec")
        support, query = data["train"], data["test"]
        label_dict = {0: "ABBR", 1: "ENTY", 2: "DESC", 3: "HUM", 4: "LOC", 5: "NUM"}
        support_meta = []
        for s in support:
            support_meta.append(
                {"question": s["text"], "answer": label_dict[s["coarse_label"]]}
            )
        query_meta = []
        for q in query:
            query_meta.append(
                {"question": q["text"], "answer": label_dict[q["coarse_label"]]}
            )
    elif dataset == "mit_movies_director":
        field_name = "Director"
        all_fields = [
            "Actor",
            "Award",
            "Character_Name",
            "Director",
            "Genre",
            "Opinion",
            "Origin",
            "Plot",
            "Quote",
            "Relationship",
            "Soundtrack",
            "Year",
        ]
        assert field_name in all_fields
        all_fields.remove(field_name)
        filter_tags = (
            [f"B-{field}" for field in all_fields]
            + [f"I-{field}" for field in all_fields]
            + ["O"]
        )
        target_tags = [f"B-{field_name}", f"I-{field_name}"]

        with open(f"{dataDir}/{dataset}/train", "r") as f:
            lines = f.readlines()
            lines = [line.replace(" <=> <NULL>", "").strip() for line in lines]
        support_meta = []
        for line in lines:
            answer = ""
            untagged_line = ""
            for word in line.split(" "):
                contains_target = [tag in word for tag in target_tags]
                if np.any(contains_target):
                    for tag in target_tags:
                        word = word.replace(":" + tag, "")
                    answer += word + " "
                for tag in filter_tags:
                    word = word.replace(":" + tag, "")
                untagged_line += word + " "

            if answer != "":
                support_meta.append(
                    {"question": untagged_line.strip(), "answer": answer.strip()}
                )

        query_meta = []
        with open(f"{dataDir}/{dataset}/test", "r") as f:
            lines = f.readlines()
            lines = [line.replace(" <=> <NULL>", "").strip() for line in lines]

        for line in lines:
            answer = ""
            untagged_line = ""
            for word in line.split(" "):
                contains_target = [tag in word for tag in target_tags]
                if np.any(contains_target):
                    for tag in target_tags:
                        word = word.replace(":" + tag, "")
                    answer += word + " "
                for tag in filter_tags:
                    word = word.replace(":" + tag, "")
                untagged_line += word + " "

            if answer != "":
                query_meta.append(
                    {"question": untagged_line.strip(), "answer": answer.strip()}
                )
    elif dataset in [
        "open_mi_captioned",
        "open_fvqa_captioned",
        "math_induction_text",
        "math_induction_text_interleaved",
        "clevr_simple_text",
        "cobsat_text",
        "open_t2i_mi_text",
        "matching_mi_text",
        "matching_mi_2_text",
    ]:
        with open(f"{dataDir}/{dataset}/query.json", "r") as f:
            query_meta = json.load(f)
        with open(f"{dataDir}/{dataset}/support.json", "r") as f:
            support_meta = json.load(f)
    return query_meta, support_meta


def encode_image(image_path):
    _, file_extension = os.path.splitext(image_path)
    file_extension = file_extension.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
    }
    mime_type = mime_types.get(file_extension)
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_image, mime_type


def summarise_results_to_csv(
    dataset, base_dir="results/", uncertainty_measure="perplexity"
):
    # Add uncertainty measure to the base directory
    base_dir = os.path.join(base_dir, uncertainty_measure)

    dataset_path = os.path.join(base_dir, dataset)

    model_files = [f for f in os.listdir(dataset_path) if f.endswith(".json")]

    # Remove results for otter-mpt
    model_files = [f for f in model_files if "otter-mpt" not in f]

    # Initialize an empty dictionary for the metrics
    data_dict = {}
    # Define the shots and metrics for establishing DataFrame structure
    all_metrics = ["acc", "auroc", "auprc", "prr"]
    if dataset == "open_mi":
        all_shots = ["0-shot", "1-shot", "2-shot", "4-shot", "5-shot"]
    else:
        all_shots = ["0-shot", "1-shot", "2-shot", "4-shot", "8-shot"]

    # Initialize an empty DataFrame with a MultiIndex for columns
    columns = pd.MultiIndex.from_product(
        [all_metrics, all_shots], names=["Metric", "Shot"]
    )
    results_df = pd.DataFrame(columns=columns)

    for file in model_files:
        file_path = os.path.join(dataset_path, file)
        with open(file_path, "r") as f:
            json_data = json.load(f)

        # Extract the model name and shot info from the file name
        model_name, shot_info = file.split("_")[0], file.split("_")[-1].replace(
            ".json", ""
        )

        # If the model name is not in the dictionary, initialize a sub-dictionary for it
        if model_name not in data_dict:
            data_dict[model_name] = {
                shot: {metric: None for metric in all_metrics} for shot in all_shots
            }

        # Update the sub-dictionary with the calibration metrics for the corresponding shot
        misclassification_metrics = json_data["misclassification_metrics"]
        for metric in all_metrics:
            # Assume the JSON structure has the metric names in lowercase
            data_dict[model_name][shot_info][metric] = misclassification_metrics.get(
                metric, None
            )

    # Populate the DataFrame
    for model, shots_data in data_dict.items():
        for shot, metrics_data in shots_data.items():
            for metric, values in metrics_data.items():
                mean = np.mean(values)
                se = np.std(values, ddof=1) / np.sqrt(len(values))
                results_df.loc[model, (metric, shot)] = f"{mean:.4f} ± {se:.4f}"

    # Reorder the columns to group by metrics first
    results_df = results_df.sort_index(axis=1, level="Metric")

    # Now save to CSV in the dataset directory
    results_csv_path = os.path.join(dataset_path, f"eval_matrics.csv")
    results_df.to_csv(results_csv_path)
    print(f"Saved CSV for {dataset} at: {results_csv_path}")


def plot_metrics(metrics, dataset, uncertainty_measure):
    # Adjusting the figure for a 2x2 layout
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"{dataset} - {uncertainty_measure} Metrics")

    # Flatten the axs from a 2x2 grid to a 1D array for easier indexing
    axs = axs.flatten()

    # Mapping metrics to the correct plot indices in a 2x2 layout
    plot_indices = {"acc": 0, "auprc": 1, "auroc": 2, "prr": 3}

    for metric, index in plot_indices.items():
        if metric in metrics.get(
            next(iter(metrics)), {}
        ):  # Check if the metric exists in data
            axs[index].set_title(metric.capitalize())
            axs[index].set_xlabel("No. of shots")
            axs[index].set_ylabel(metric.upper())

            for engine in metrics:
                if metric in metrics[engine]:
                    shots, values_list = zip(*metrics[engine][metric])
                    # Convert shots to integers
                    shots = [int(shot) for shot in shots]
                    means = [np.mean(values) for values in values_list]
                    std_errors = [
                        np.std(values, ddof=1) / np.sqrt(len(values))
                        for values in values_list
                    ]

                    axs[index].errorbar(
                        shots,
                        means,
                        yerr=std_errors,
                        label=engine,
                        marker="o",
                        capsize=3,
                    )

            axs[index].set_xticks(sorted(shots))
            axs[index].legend()

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the rectangle in tight_layout
    plt.savefig(
        f"results/{uncertainty_measure}/{dataset}/misclassification_metrics_{dataset}_{uncertainty_measure}.pdf"
    )
    plt.show()


def plot_uncertainty_correctness_separability(
    data, dataset, engines, n_shots, uncertainty_measure
):
    # Define the size of the grid
    fig, axs = plt.subplots(
        len(engines), len(n_shots), figsize=(15, 10), sharex="col", sharey="row"
    )
    fig.suptitle(f"{dataset}")

    # Loop over engines and shots to plot data
    for i, engine in enumerate(engines):
        for j, shot in enumerate(n_shots):
            ax = axs[i][j] if len(engines) > 1 else axs[j]
            # Collect data for the current engine and shot
            correct_scores, incorrect_scores = [], []
            for entry in data[engine][shot]:
                for idx, label in enumerate(entry["label"]):
                    if label == 1:
                        correct_scores.append(entry["output_uncertainty"][idx])
                    else:
                        incorrect_scores.append(entry["output_uncertainty"][idx])

            # Define bins for histograms
            min_score = min(correct_scores + incorrect_scores)
            max_score = max(correct_scores + incorrect_scores)
            bins = np.linspace(min_score, max_score, 20)

            # Plot histograms
            ax.hist(correct_scores, bins, alpha=0.5, label="Correct", color="green")
            ax.hist(incorrect_scores, bins, alpha=0.5, label="Incorrect", color="red")
            ax.set_title(f"{shot}-shot" if i == 0 else "")
            ax.set_ylabel(f"{engine}" if j == 0 else "")
            ax.legend()

    # Adjust layout
    plt.tight_layout()
    plt.savefig(
        f"results/{uncertainty_measure}/{dataset}/uncertainty_correctness_separability_{dataset}_{uncertainty_measure}.pdf"
    )
    plt.show()


def load(data_dir, dataset):
    """Load the logos dataset and organize the data by background type and logo for OCR.
    Args:
        data_dir: The directory containing the logos dataset.
    Returns:
        dict: A dictionary containing the paths to the logo images organized by background type and then by logo.
    """
    if "logos" in dataset:
        classes = LOGOS
    elif "sin" in dataset:
        classes = SIN
    elif "icons" in dataset:
        classes = ICONS

    objects = {}  # Dictionary to store logos organized by background and then by logo

    # List prompt types to consolidate
    prompt_types = ["simple_prompts", "complex_prompts"]

    # Iterate over both 'simple_prompts' and 'complex_prompts' directories
    for prompt_type in prompt_types:
        prompt_directory = os.path.join(data_dir, prompt_type)

        for object in classes:
            for type_label in ["Easy", "Hard"]:
                if "logos" in dataset:
                    object_dir = object
                elif "sin" in dataset:
                    object_dir = object[0].upper() + object[1:]
                elif "icons" in dataset:
                    if object == "face_emoji":
                        object_dir = (
                            object[0].upper()
                            + object[1:5]
                            + object[5].upper()
                            + object[6:]
                        )
                    else:
                        object_dir = object[0].upper() + object[1:]
                else:
                    raise ValueError("Dataset not found!")
                directory = os.path.join(prompt_directory, object_dir, type_label)

                # Safely attempt to list files in the directory
                try:
                    file_names = os.listdir(directory)
                except FileNotFoundError:
                    print(f"Here, Directory not found: {directory}")
                    continue  # Skip this directory if not found

                for file_name in file_names:
                    if file_name.endswith(".png"):  # Process only PNG files
                        parts = file_name.split("-")
                        if "icons" in dataset:
                            background = parts[2]
                        else:
                            background = parts[
                                1
                            ]  # Assuming background is the second part of the file name

                        # Ensure storage structures exist for backgrounds
                        if background not in objects:
                            objects[background] = {}
                        if object not in objects[background]:
                            objects[background][object] = []

                        # Append the file path to the appropriate category
                        objects[background][object].append(
                            os.path.join(directory, file_name)
                        )

    return objects  # Return the fully populated dictionary after processing all files


def load_messages(data_dir):
    """Load the logos dataset and organize the data by background type and logo for OCR.
    Args:
        data_dir: The directory containing the logos dataset.
    Returns:
        dict: A dictionary containing the paths to the logo images organized by background type and then by logo.
    """
    classes = MESSAGES

    objects = {}  # Dictionary to store logos organized by background and then by logo

    # Safely attempt to list files in the directory
    try:
        file_names = os.listdir(data_dir)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Directory not found!"
        )  # Skip this directory if not found

    for file_name in file_names:
        if file_name.endswith(".png"):  # Process only PNG files
            parts = file_name.split("-")

            background = parts[1]

            object = parts[0]

            # Ensure storage structures exist for backgrounds
            if background not in objects:
                objects[background] = {}
            if object not in objects[background]:
                objects[background][object] = []

            # Append the file path to the appropriate category
            objects[background][object].append(os.path.join(data_dir, file_name))

    return objects  # Return the fully populated dictionary after processing all files


def load_logo_conditioning(dataset):
    """Load the logos dataset and organize the data by background type and logo for OCR.
    Args:
        data_dir: The directory containing the logos dataset.
    Returns:
        dict: A dictionary containing the paths to the logo images organized by prompt type, background type, and then by logo.
    """

    if dataset == "logos_conditioning":
        data_dir = CONDITIONING_IMAGES_LOGOS
    elif dataset == "sin_conditioning":
        data_dir = CONDITIONING_IMAGES_SIN
    elif dataset == "icons_conditioning":
        data_dir = CONDITIONING_IMAGES_ICONS

    objects = {}  # Use dictionaries for simple and complex prompts

    # Loop over the folders in this directory
    for folder in os.listdir(data_dir):
        # Each folder is relates to a logo.
        if folder == ".DS_Store":
            continue
        obj = folder.lower()
        objects[obj] = []
        for file_name in os.listdir(os.path.join(data_dir, folder)):
            if file_name.endswith(".png"):  # Process only PNG files
                # Append the file path to the appropriate category
                objects[obj].append(os.path.join(data_dir, folder, file_name))

    return objects  # Return the fully populated dictionary after processing all files
