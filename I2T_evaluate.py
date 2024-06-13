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


def parse_args():
    parser = argparse.ArgumentParser(description="I2T ICL Evaluation")

    parser.add_argument(
        "--dataDir", default="./VL-ICL", type=str, help="Data directory."
    )
    parser.add_argument(
        "--dataset",
        default="operator_induction",
        type=str,
        choices=[
            "operator_induction",
            "textocr",
            "open_mi",
            "clevr",
            "operator_induction_interleaved",
            "matching_mi",
            "logos_neither",
            "logos_logo",
            "logos_background",
            "logos_both",
            "logos_conditioning",
            "sin_conditioning",
            "logos_background_neither",
            "logos_background_logo",
            "logos_background_background",
            "logos_background_both",
            "logos_both_neither",
            "logos_both_logo",
            "logos_both_background",
            "logos_both_both",
            "sin_neither",
            "sin_object",
            "sin_background",
            "sin_both",
            "sin_background_neither",
            "sin_background_object",
            "sin_background_background",
            "sin_background_both",
            "sin_both_neither",
            "sin_both_object",
            "sin_both_background",
            "sin_both_both",
            "icons_neither",
            "icons_icon",
            "icons_background",
            "icons_both",
            "icons_background_neither",
            "icons_background_icon",
            "icons_background_background",
            "icons_background_both",
            "icons_both_neither",
            "icons_both_icon",
            "icons_both_background",
            "icons_both_both",
            "icons_conditioning",
            "messages_conditioning",
            "messages_neither",
            "messages_message",
            "messages_background",
            "messages_both",
            "messages_background_neither",
            "messages_background_message",
            "messages_background_background",
            "messages_background_both",
            "messages_both_neither",
            "messages_both_message",
            "messages_both_background",
        ],
    )
    parser.add_argument(
        "--engine",
        "-e",
        choices=[
            "openflamingo",
            "otter-llama",
            "otter-mpt",
            "llava16-7b",
            "qwen-vl",
            "qwen-vl-chat",
            "internlm-x2",
            "emu2-chat",
            "idefics-9b-instruct",
            "idefics-80b-instruct",
            "gpt4v",
            "mmicl-t5-xl",
            "mmicl-t5-xxl",
        ],
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
