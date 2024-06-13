import numpy as np
import re
from utils.constants import LOGOS, LOGOS_TASKS, MODELS, NUM_RERUNS, SIN, ICONS
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import re


def eval_scores(results, dataset):
    """Evaluate the model on the given dataset.
    Args:
        results: List of results from the model.
        dataset: Name of the dataset.

    Returns:
        eval_scores: Dictionary of evaluation metrics.
    """
    if dataset == "matching_mi":
        # Accuracy based on the exact match of the prediction and the answer.
        accs = exact_yes_no(results)
    else:
        # Accuracy based on of the answer occurs in the prediction.
        accs = exact_match(results, dataset)

    return {
        "acc": accs,
    }


def normalize_text(text):
    # Remove punctuation and convert to lower case for uniformity
    text = re.sub(r"[^\w\s]", "", text)
    text = text.replace("_", " ")

    return text.lower()


def eval_cls_and_background_accuracy(results, dataset):

    # Initialize variables to compute accuracy for result_cls and background
    total = 0
    background_correct = 0
    cls_correct = 0

    # Process each result entry
    for result in results:
        if "logos" in dataset:
            result_cls = result["logo"]
        elif "sin" in dataset:
            result_cls = result["object"]
        elif "icons" in dataset:
            result_cls = result["icon"]
        elif "messages" in dataset:
            result_cls = result["message"]
        else:
            raise ValueError("Dataset not recognized")

        background = result["background"]

        result_cls = normalize_text(result_cls)
        background = normalize_text(background)

        predictions = result["prediction"]

        # Ensure predictions is a list, if not, convert it to a list with one item
        if not isinstance(predictions, list):
            predictions = [predictions]

        # Only consider the first item in the list of predictions
        prediction = normalize_text(predictions[0])

        # Truncate prediction at the first newline or period
        trunc_index = prediction.find("\n")
        if trunc_index <= 0:
            trunc_index = prediction.find(".")
        if trunc_index > 0:
            prediction = prediction[:trunc_index]

        # Check if the normalized result_cls is within the normalized prediction
        cls_is_correct = int(result_cls in prediction)
        cls_correct += cls_is_correct

        # Check if the normalized background is within the normalized prediction
        background_is_correct = int(background in prediction)
        background_correct += background_is_correct

        total += 1

    # Calculate accuracy for result_cls
    cls_accuracy = cls_correct / total if total > 0 else 0

    # Calculate accuracy for background
    background_accuracy = background_correct / total if total > 0 else 0

    # Return results as a dictionary with result_cls accuracy and background accuracy
    return {
        "shape_acc": cls_accuracy,
        "background_acc": background_accuracy,
    }


# This function now calculates accuracies for each of the multiple predictions per result, tracking the performance across multiple runs or iterations.


def exact_yes_no(results):
    """Calculate accuracy for each rerun based on exact match between multiple predictions and the true answer, and
    append a 'label' list to each result dict indicating the correctness of each prediction.

    Args:
        results: List of dictionaries, each containing a list of 'prediction' strings and a single 'answer' string.

    Returns:
        avg_accs: List of average accuracies, one for each rerun.
    """
    # Determine the number of reruns based on the length of the first result's prediction list.
    if not results:
        return []

    num_reruns = len(results[0]["prediction"])

    # Initialize a list to collect accuracies for each rerun.
    rerun_accuracies = [[] for _ in range(num_reruns)]

    # Process each result entry.
    for result in results:
        answer = result["answer"]
        predictions = result["prediction"]
        result["label"] = []  # Initialize the label list for this result

        # Process each prediction in the list corresponding to reruns.
        for index, prediction in enumerate(predictions):
            prediction = prediction.strip().strip("\n")
            if prediction == answer:
                rerun_accuracies[index].append(1)
                result["label"].append(1)
            else:
                rerun_accuracies[index].append(0)
                result["label"].append(0)

    # Calculate average accuracy for each rerun.
    avg_accs = [
        np.mean(rerun_acc) if rerun_acc else 0 for rerun_acc in rerun_accuracies
    ]

    return avg_accs


def exact_in_match(results):
    # Determine the number of reruns from the first result's prediction list length
    num_reruns = len(results[0]["prediction"])

    # Initialize a list to collect accuracies for each rerun
    rerun_accuracies = [[] for _ in range(num_reruns)]

    # Process each result entry
    for result in results:
        answers = result["answer"]
        predictions = result["prediction"]
        result["label"] = []  # Initialize the label list for this result

        # Process each prediction in the list corresponding to reruns
        for index, prediction in enumerate(predictions):
            prediction = prediction.strip().strip("\n")
            trunc_index = prediction.find("\n")
            if trunc_index <= 0:
                trunc_index = prediction.find(".")
            if trunc_index > 0:
                prediction = prediction[:trunc_index]

            # Check if the answer is within the prediction, considering the dataset adjustments
            if str(answers).lower() in prediction.lower():
                rerun_accuracies[index].append(1)
                result["label"].append(1)
            else:
                rerun_accuracies[index].append(0)
                result["label"].append(0)

    # Calculate average accuracy for each rerun
    avg_accs = [
        np.mean(rerun_acc) if rerun_acc else 0 for rerun_acc in rerun_accuracies
    ]

    return avg_accs


# def exact_match(results, dataset):
#     acc = []
#     for result in results:
#         prediction = result["prediction"].strip()
#         prediction = prediction.strip("\n")
#         trunc_index = prediction.find("\n")
#         if trunc_index <= 0:
#             trunc_index = prediction.find(".")
#         if trunc_index > 0:
#             prediction = prediction[:trunc_index]
#         if "operator_induction" in dataset or "clevr_simple" in dataset:
#             # find the number
#             # NOTE: Altered from the orignal code to capture negative numbers too.
#             match = re.search(r"\d+", prediction)
#             if match:
#                 prediction = match.group()
#             else:
#                 prediction = ""

#         if str(prediction).lower() == str(result["answer"]).lower():
#             acc.append(1)
#             result["label"] = 1
#         else:
#             acc.append(0)
#             result["label"] = 0

#         # If a classification task, the label is just the answer, standard calibrations.
#         if dataset in CLASSIFICATION_TASKS:
#             result["label"] = result["answer"]

#     avg_acc = np.average(acc)
#     return avg_acc


def exact_match(results, dataset):
    # Determine the number of reruns from the first result's prediction list length
    num_reruns = len(results[0]["prediction"])

    # Initialize a list to collect accuracies for each rerun
    rerun_accuracies = [[] for _ in range(num_reruns)]

    # Process each result entry
    for result in results:
        answers = result["answer"]
        predictions = result["prediction"]
        result["label"] = []  # Initialize the label list for this result

        # Process each prediction in the list corresponding to reruns
        for index, prediction in enumerate(predictions):
            prediction = prediction.strip()
            prediction = prediction.strip("\n")
            trunc_index = prediction.find("\n")
            if trunc_index <= 0:
                trunc_index = prediction.find(".")
            if trunc_index > 0:
                prediction = prediction[:trunc_index]

            # Handle specific datasets by extracting the number if applicable
            if "operator_induction" in dataset or "clevr_simple" in dataset:
                # NOTE: Altered from the original code to capture negative numbers too.
                match = re.search(r"-?\d+", prediction)
                if match:
                    prediction = match.group()
                else:
                    prediction = ""

            # Check if the answer is within the prediction, considering the dataset adjustments
            if str(answers).lower() == prediction.lower():
                rerun_accuracies[index].append(1)
                result["label"].append(1)
            else:
                rerun_accuracies[index].append(0)
                result["label"].append(0)

    # Calculate average accuracy for each rerun
    avg_accs = [
        np.mean(rerun_acc) if rerun_acc else 0 for rerun_acc in rerun_accuracies
    ]

    return avg_accs


def plot_accuracy():
    # Directory structure information
    base_dir = "results"
    datasets = LOGOS_TASKS
    models = MODELS
    shots = [0, 1, 2, 4, 8]

    # Initialize a dictionary to store the data
    data = {dataset: pd.DataFrame(index=shots) for dataset in datasets}

    # Load and process each JSON file
    for dataset in datasets:
        for model in models:
            accuracies = []
            for shot in shots:
                file_path = f"{base_dir}/{dataset}/{model}_{shot}-shot.json"
                if os.path.exists(file_path):
                    with open(file_path, "r") as file:
                        json_data = json.load(file)
                        correct_labels = [entry["correct_label"] for entry in json_data]
                        accuracy = sum(correct_labels) / len(correct_labels)
                        accuracies.append(accuracy)
                else:
                    accuracies.append(None)
            data[dataset][model] = accuracies

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    fig.suptitle("Logo Prediction")

    for ax, (dataset, df) in zip(axs.flatten(), data.items()):
        df.plot(ax=ax, marker="o", linestyle="-")
        ax.set_title(dataset)
        ax.set_xticks(shots)
        ax.set_xlabel("Number of Shots")
        ax.set_ylabel("Accuracy")
        ax.grid(True)

    # Tight layout with space for the supertitle
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
