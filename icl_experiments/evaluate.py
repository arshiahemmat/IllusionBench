import json
import argparse
from evals import eval
from collections import defaultdict
from utils.constants import MODELS, ALL_TASKS


def parse_args():
    """Parses command-line arguments for I2T ICL evaluation."""
    parser = argparse.ArgumentParser(description="I2T ICL Evaluation")

    # Directory containing the results and data files
    parser.add_argument(
        "--dataDir", default="./VL-ICL", type=str, help="Data directory."
    )
    
    # Dataset to evaluate
    parser.add_argument(
        "--dataset",
        default="logos_tc_icl_both",
        type=str,
        choices=ALL_TASKS,
        help="Dataset to evaluate.",
    )
    
    # Model engines to evaluate
    parser.add_argument(
        "--engine",
        "-e",
        choices=MODELS,
        default=["llava16-7b"],
        nargs="+",
        help="Model engine choices.",
    )
    ƒ
    # Number of support images (shots) to evaluate
    parser.add_argument(
        "--n_shot",
        default=[0, 1, 2, 4, 8],
        nargs="+",
        help="Number of support images (shots).",
    )

    return parser.parse_args()


def process_engine_results(engine, args):
    """
    Processes the evaluation results for a given engine across different shots.

    Args:
        engine (str): The model engine to evaluate.
        args (Namespace): Parsed command-line arguments.

    Returns:
        engine_results (dict): Dictionary of shape and scene accuracies for each shot.
    """
    engine_results = {}

    # Determine the shots to evaluate based on the dataset.
    if "icons" in args.dataset:
        args.n_shot = [0, 1, 2, 4, 5]
    else:
        args.n_shot = [0, 1, 2, 4, 8]

    # Evaluate shape and scene accuracy for each shot.
    for shot in args.n_shot:
        result_file = f"results/{args.dataset}/{engine}_{shot}-shot.json"

        # Load the results from the JSON file for the current shot.
        with open(result_file, "r") as f:
            original_data = json.load(f)

        # Evaluate shape and scene accuracy using the eval function.
        results = eval.eval_shape_and_scene_accuracy(original_data, args.dataset)

        # Store the accuracies in the results dictionary.
        engine_results[shot] = {
            "shape_accuracy": results["shape_acc"],
            "scene_accuracy": results["scene_acc"],
        }

    return engine_results


if __name__ == "__main__":
    args = parse_args()

    # Initialize metrics and individual data storage.
    metrics = defaultdict(lambda: defaultdict(list))
    individual_data = defaultdict(lambda: defaultdict(list))

    # Dictionary to store the evaluation results for all engines.
    all_engine_results = {}

    # Process evaluation results for each engine specified in the arguments.
    for engine in args.engine:
        engine_results = process_engine_results(engine, args)

        # Store the results for the current engine.
        all_engine_results[engine] = engine_results

    # Save the evaluation results to a JSON file.
    output_file = f"results/{args.dataset}/results.json"
    with open(output_file, "w") as f:
        json.dump(all_engine_results, f, indent=4)