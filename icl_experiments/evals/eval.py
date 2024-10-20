import re

def normalize_text(text):
    """
    Normalize text by removing punctuation, replacing underscores, 
    and converting to lowercase.
    """
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    return text.replace("_", " ").lower()


def truncate_prediction(prediction):
    """
    Truncate prediction at the first newline or period.
    """
    trunc_index = prediction.find("\n")
    if trunc_index == -1:  # If no newline, look for a period
        trunc_index = prediction.find(".")
    if trunc_index > 0:
        return prediction[:trunc_index].strip()
    return prediction.strip()


def get_result_class(result, dataset):
    """
    Extract the correct class (shape or object) from the result based on the dataset.
    """
    dataset_mapping = {
        "logos": "logo",
        "in": "object",
        "icons": "icon",
        "messages": "message"
    }

    for key, value in dataset_mapping.items():
        if key in dataset:
            return result[value]
    
    raise ValueError("Dataset not recognized")


def eval_shape_and_scene_accuracy(results, dataset):
    """
    Evaluate the accuracy of shape (object) and scene predictions from the results.
    """
    total = len(results)
    scene_correct = 0
    shape_correct = 0

    # Iterate through each result entry and evaluate predictions
    for result in results:
        # Extract shape and scene from result
        shape = normalize_text(result.get("shape", ""))
        scene = normalize_text(result.get("scene", ""))

        # Get the predictions, ensure it's a list
        predictions = result.get("prediction", [])
        if not isinstance(predictions, list):
            predictions = [predictions]

        # Normalize and truncate the first prediction for comparison
        prediction = normalize_text(truncate_prediction(predictions[0]))

        # Check if the predicted shape and scene are correct
        shape_correct += int(shape in prediction)
        scene_correct += int(scene in prediction)

    # Calculate accuracy for shape (object) and scene
    shape_acc = shape_correct / total if total > 0 else 0
    scene_acc = scene_correct / total if total > 0 else 0

    return {
        "shape_acc": shape_acc,
        "scene_acc": scene_acc,
    }
