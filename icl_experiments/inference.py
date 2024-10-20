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
from utils.constants import TASKS, ALL_TASKS, MODELS, NUM_RERUNS, LOGOS, IN, ICONS, LOGOS_DATA_FOLDER, IN_DATA_FOLDER, ICON_DATA_FOLDER
from utils.utils import load, load_logo_conditioning


def parse_args():
    """Parses command-line arguments for I2T ICL inference."""
    parser = argparse.ArgumentParser(description="I2T ICL Inference")

    parser.add_argument(
        "--dataDir",
        default="/scratch/local/ssd/tomlamb/vlm-icl",
        type=str,
        help="Data directory.",
    )
    parser.add_argument(
        "--dataset",
        default=["logos_tc_icl_both"],
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
        help="Model engine choices.",
    )
    parser.add_argument(
        "--n_shot",
        default=[0, 1, 2, 4, 8],
        nargs="+",
        help="Number of support images.",
    )
    parser.add_argument(
        "--max-new-tokens", 
        default=15, 
        type=int, 
        help="Max new tokens for generation."
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


def load_dataset(eval_dataset):
    """
    Loads the appropriate dataset or conditioning dataset based on eval_dataset.
    Simplifies logic by reducing redundancy.
    """
    if eval_dataset in TASKS["logos"]:
        return load(LOGOS_DATA_FOLDER, eval_dataset)
    elif eval_dataset in TASKS["in"]:
        return load(IN_DATA_FOLDER, eval_dataset)
    elif eval_dataset in TASKS["icons"]:
        return load(ICON_DATA_FOLDER, eval_dataset)
    elif eval_dataset in ["logos_conditioning", "in_conditioning", "icons_conditioning"]:
        return load_logo_conditioning(eval_dataset)
    else:
        raise ValueError(f"Unknown dataset: {eval_dataset}")


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
    """Evaluates questions based on the dataset, model, and specified shot settings."""
    results = []
    data_path = ""

    # Determine class types based on the dataset
    if "logos" in eval_dataset:
        shapes = LOGOS
    elif "in" in eval_dataset:
        shapes = IN
    elif "icons" in eval_dataset:
        shapes = ICONS
    else:
        raise ValueError(f"Unknown dataset: {eval_dataset}")

    # Load dataset based on the task
    dataset = load_dataset(eval_dataset)

    max_new_tokens = args.max_new_tokens

    # If the dataset is not a conditioning dataset, populate query and support metadata
    if "conditioning" not in eval_dataset:
        # Gather query metadata for each prompt type and scene
        for scene in dataset:
            for shape in shapes:
                for image_path in dataset[scene][shape]:
                    query = {
                        "image": image_path,
                        "scene": scene,
                        "shape": shape,
                    }
                    query_meta.append(query)

        # Filter dataset for support metadata
        for scene in dataset:
            for shape in list(dataset[scene]):
                if shape not in shapes:
                    del dataset[scene][shape]
        
        support_meta = dataset
    
    else:
        # Gather support metadata for conditioning datasets
        support_meta = []
        for shape in dataset:
            for image_path in dataset[shape]:
                query = {"image": image_path, "shape": shape}
                query_meta.append(query)

    print(f"Running inference for {n_shot}-shot on {eval_dataset} with {args.engine}...")

    # Run inference over the dataset
    for idx, query in enumerate(query_meta):
        if idx % 250 == 0:
            print("Processing query:", idx)

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

    # Main execution: Loop over datasets and engines for inference
    for eval_dataset in args.dataset:
        for engine in args.engine:

            model, tokenizer, processor = load_models.load_i2t_model(engine, args)
            print(f"Loaded model: {engine}\n")

            utils.set_random_seed(args.seed)

            # Evaluate for each specified shot setting
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
                
                # Save the results
                os.makedirs(f"results/{eval_dataset}", exist_ok=True)
                with open(f"results/{eval_dataset}/{engine}_{shot}-shot.json", "w") as f:
                    json.dump(results_dict, f, indent=4)

            # Clean up GPU memory
            del model, tokenizer, processor
            torch.cuda.empty_cache()
            gc.collect()
