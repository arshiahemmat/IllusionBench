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

    # Initialize variables to compute shape and scene accuracy.
    total = 0
    scene_correct = 0
    shape_correct = 0

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
        shape_correct += cls_is_correct

        # Check if the normalized scene is within the normalized prediction
        scene_is_correct = int(background in prediction)
        scene_correct += scene_is_correct

        total += 1

    # Calculate accuracy for result_cls
    cls_accuracy = shape_correct / total if total > 0 else 0

    # Calculate accuracy for background
    scene_accuracy = scene_correct / total if total > 0 else 0

    # Return results as a dictionary with result_cls accuracy and background accuracy
    return {
        "shape_acc": cls_accuracy,
        "scene_acc": scene_accuracy,
    }
