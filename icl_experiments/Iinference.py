import torch
import os
import json
import argparse
import gc
from utils import (
    model_inference,
    utils,
    ICL_utils,
    load_models,
)
from utils.constants import (
    LOGOS_TASKS,
    LOGOS,
    NUM_RERUNS,
    LOGOS_BACKGROUND_TASKS,
    SIN,
    SIN_DATA_FOLDER,
    SIN_TASKS,
    SIN_BACKGROUND_TASKS,
    LOGOS_DATA_FOLDER,
    ICON_DATA_FOLDER,
    ICONS_TASKS,
    ICONS_BACKGROUND_TASKS,
    ICONS,
    ICONS_BOTH_TASKS,
    SIN_BOTH_TASKS,
    LOGOS_BOTH_TASKS,
    MESSAGES,
    MESSAGES_TASKS,
    MESSAGES_BACKGROUND_TASKS,
    MESSAGES_BOTH_TASKS,
    MESSAGES_DATA_FOLDER,
    MESSAGES_DOMAINS,
    MODELS,
    ALL_TASKS,
)
from utils.utils import load, load_logo_conditioning, load_messages


def parse_args():
    parser = argparse.ArgumentParser(description="I2T ICL Inference")

    parser.add_argument(
        "--dataDir",
        default="/scratch/local/ssd/tomlamb/vlm-icl",
        type=str,
        help="Data directory.",
    )
    parser.add_argument(
        "--dataset",
        default=["operator_induction"],
        type=str,
        nargs="+",
        choices=ALL_TASKS,
        help="List of datasets.",
    )
    parser.add_argument(
        "--engine",
        "-e",
        choices=MODELS,
        default=["llava16-7b"],
        nargs="+",
    )
    parser.add_argument(
        "--n_shot",
        default=[0, 1, 2, 4, 8],
        nargs="+",
        help="Number of support images.",
    )

    parser.add_argument(
        "--max-new-tokens", default=15, type=int, help="Max new tokens for generation."
    )
    parser.add_argument(
        "--task_description",
        default="nothing",
        type=str,
        choices=["nothing", "concise", "detailed"],
        help="Detailed level of task description.",
    )

    parser.add_argument(
        "--engine_device",
        default="cuda:0",
        type=str,
        help="Device to run the engine on.",
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    return parser.parse_args()


def eval_questions(
    args,
    eval_dataset,
    query_meta,
    support_meta,
    model,
    tokenizer,
    processor,
    engine,
    n_shot,
):
    results = []
    data_path = ""

    if "logos" in eval_dataset:
        classes = LOGOS
    elif "sin" in eval_dataset:
        classes = SIN
    elif "icons" in eval_dataset:
        classes = ICONS
    elif "messages" in eval_dataset:
        classes = MESSAGES
    else:
        raise ValueError("Unknown dataset: {}".format(eval_dataset))

    if (
        eval_dataset
        in LOGOS_TASKS
        + LOGOS_BACKGROUND_TASKS
        + LOGOS_BOTH_TASKS
        + SIN_TASKS
        + SIN_BACKGROUND_TASKS
        + SIN_BOTH_TASKS
        + ICONS_TASKS
        + ICONS_BACKGROUND_TASKS
        + ICONS_BOTH_TASKS
        + MESSAGES_TASKS
        + MESSAGES_BACKGROUND_TASKS
        + MESSAGES_BOTH_TASKS
    ):
        # Load the logos dataset organised for background.
        if eval_dataset in LOGOS_TASKS + LOGOS_BACKGROUND_TASKS + LOGOS_BOTH_TASKS:
            dataset = load(LOGOS_DATA_FOLDER, eval_dataset)
        elif eval_dataset in SIN_TASKS + SIN_BACKGROUND_TASKS + SIN_BOTH_TASKS:
            dataset = load(SIN_DATA_FOLDER, eval_dataset)
        elif eval_dataset in ICONS_TASKS + ICONS_BACKGROUND_TASKS + ICONS_BOTH_TASKS:
            dataset = load(ICON_DATA_FOLDER, eval_dataset)
        elif (
            eval_dataset
            in MESSAGES_TASKS + MESSAGES_BACKGROUND_TASKS + MESSAGES_BOTH_TASKS
        ):
            dataset = load_messages(MESSAGES_DATA_FOLDER, eval_dataset)

    elif (
        # Load condiitonig dataset for inital dataset exploration.
        eval_dataset == "logos_conditioning"
        or eval_dataset == "sin_conditioning"
        or eval_dataset == "icons_conditioning"
        or eval_dataset == "messages_conditioning"
    ):
        dataset = load_logo_conditioning(eval_dataset)
    else:
        raise ValueError("Unknown dataset: {}".format(eval_dataset))

    max_new_tokens = args.max_new_tokens

    if "conditioning" not in eval_dataset:
        # Iterate over each prompt type and background in the dataset
        for background in dataset:
            for cls in classes:
                for image_path in dataset[background][cls]:
                    # Create a query object for each image
                    query = {
                        "image": image_path,
                        "background": background,
                    }
                    if "logos" in eval_dataset:
                        query["logo"] = cls
                    elif "sin" in eval_dataset:
                        query["object"] = cls
                    elif "icons" in eval_dataset:
                        query["icon"] = cls
                    elif "messages" in eval_dataset:
                        query["message"] = cls
                    query_meta.append(query)

        # Need to loop over the dataset and only include shapes in dataset in the support set.
        for background in dataset:
            for cls in dataset[background]:
                if cls not in classes:
                    del dataset[background][cls]

        support_meta = dataset
    else:
        support_meta = []
        for object in dataset:
            for image_path in dataset[object]:
                if eval_dataset == "logos_conditioning":
                    query = {"image": image_path, "logo": object}
                elif eval_dataset == "sin_conditioning":
                    query = {"image": image_path, "object": object}
                elif eval_dataset == "icons_conditioning":
                    query = {"image": image_path, "icon": object}
                elif eval_dataset == "messages_conditioning":
                    query = {"image": image_path, "message": object}
                query_meta.append(query)

    print(f"Running inference for {n_shot} on {eval_dataset} for {args.engine}...")

    for idx, query in enumerate(query_meta):
        if idx % 250 == 0:
            print("Processing query: ", idx)

        predictions = []
        for _ in range(NUM_RERUNS):
            n_shot_support = ICL_utils.select_demonstration(
                support_meta,
                n_shot,
                eval_dataset,
                query=query,
            )

            predicted_answer = model_inference.icl_inference(
                args,
                engine,
                eval_dataset,
                model,
                tokenizer,
                query,
                n_shot_support,
                data_path,
                processor,
                max_new_tokens,
            )

            predictions.append(predicted_answer)

        query["prediction"] = predictions
        results.append(query)

    return results


if __name__ == "__main__":
    args = parse_args()

    # Okay first we need to load the dataset into a datastructure for better access.

    # For the logos dataset, we don't have seperate files. We just have the folder.
    for eval_dataset in args.dataset:
        for engine in args.engine:

            model, tokenizer, processor = load_models.load_i2t_model(engine, args)
            print("Loaded model: {}\n".format(engine))

            utils.set_random_seed(args.seed)
            for shot in args.n_shot:
                query_meta, support_meta = [], []
                results_dict = eval_questions(
                    args,
                    eval_dataset,
                    query_meta,
                    support_meta,
                    model,
                    tokenizer,
                    processor,
                    engine,
                    int(shot),
                )
                os.makedirs(f"results/{eval_dataset}", exist_ok=True)
                with open(
                    f"results/{eval_dataset}/{engine}_{shot}-shot.json",
                    "w",
                ) as f:
                    json.dump(results_dict, f, indent=4)

            del model, tokenizer, processor
            torch.cuda.empty_cache()
            gc.collect()
