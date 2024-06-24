import json
import argparse
from evals import eval
from utils.utils import (
    summarise_results_to_csv,
    plot_uncertainty_correctness_separability,
    plot_metrics,
)
from collections import defaultdict
import numpy as np
from utils.constants import MODELS, ALL_TASKS


def parse_args():
    parser = argparse.ArgumentParser(description="I2T ICL Evaluation")

    parser.add_argument(
        "--dataDir", default="./VL-ICL", type=str, help="Data directory."
    )
    parser.add_argument(
        "--dataset",
        default="operator_induction",
        type=str,
        choices=ALL_TASKS,
    )
    parser.add_argument(
        "--engine",
        "-e",
        choices=MODELS,
        default=[
            "otter-mpt",
            "llava16-7b",
            "qwen-vl-chat",
            "idefics-9b-instruct",
            "mmicl-t5-xxl",
        ],
        nargs="+",
    )
    parser.add_argument(
        "--n_shot",
        default=[0, 1, 2, 4, 8, 10],
        nargs="+",
        help="Number of support images.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Summarise the results to a csv file in tabular form.

    metrics = defaultdict(lambda: defaultdict(list))

    individual_data = defaultdict(lambda: defaultdict(list))

    engine_results = {}

    for engine in args.engine:
        # Set the shots depending on which dataset that we are considering.
        if "icons" in args.dataset:
            args.n_shot = [0, 1, 2, 4, 5]
        else:
            args.n_shot = [0, 1, 2, 4, 8]

        # Cmopute shape and background accuracy for each engine and shot.
        for shot in args.n_shot:
            result_file = f"results/{args.dataset}/{engine}_{shot}-shot.json"
            with open(result_file, "r") as f:
                original_data = json.load(f)

            results = eval.eval_cls_and_background_accuracy(original_data, args.dataset)

            if engine not in engine_results:
                engine_results[engine] = {}

            engine_results[engine][shot] = {
                "shape_accuracy": results["shape_acc"],
                "background_accuracy": results["background_acc"],
            }

    with open(f"results/{args.dataset}/results.json", "w") as f:
        json.dump(engine_results, f, indent=4)
