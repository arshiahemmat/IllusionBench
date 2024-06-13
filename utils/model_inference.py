import torch

try:
    from llava.conversation import conv_templates
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.mm_utils import tokenizer_image_token
except:
    pass

import os
import time
from PIL import Image
from .ICL_utils import get_task_instruction, format_answer
from .utils import load_image, encode_image
import torch.nn.functional as F
from .constants import (
    LOGOS_TASKS,
    LOGOS_BACKGROUND_TASKS,
    SIN_TASKS,
    SIN_BACKGROUND_TASKS,
    ICONS_TASKS,
    ICONS_BACKGROUND_TASKS,
    LOGOS_BOTH_TASKS,
    SIN_BOTH_TASKS,
    ICONS_BOTH_TASKS,
    MESSAGES_TASKS,
    MESSAGES_BOTH_TASKS,
    MESSAGES_BACKGROUND_TASKS,
)
from itertools import combinations
import numpy as np
from matplotlib import pyplot as plt

DATASETS = (
    LOGOS_TASKS
    + SIN_TASKS
    + LOGOS_BACKGROUND_TASKS
    + SIN_BACKGROUND_TASKS
    + ICONS_TASKS
    + ICONS_BACKGROUND_TASKS
    + ICONS_BOTH_TASKS
    + SIN_BOTH_TASKS
    + LOGOS_BOTH_TASKS
    + MESSAGES_TASKS
    + MESSAGES_BOTH_TASKS
    + MESSAGES_BACKGROUND_TASKS
)


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


@torch.no_grad()
def icl_inference(
    args,
    engine,
    dataset,
    model,
    tokenizer,
    query,
    n_shot_support,
    data_path,
    processor,
    max_new_tokens,
):

    task_instruction = get_task_instruction(dataset, args)
    img_id = query["image"]
    query_images, query_image_paths = load_image(img_id, data_path)

    print("Query image",img_id)
    for i in range(len(n_shot_support)):
        print("Support image",n_shot_support[i]["image"])
    print("__________")

    if dataset not in DATASETS and "conditioning" not in dataset:
        query_text = query["question"]

    if "qwen-vl" in engine:
        # Inference routine for Qwen-VL model.

        # Prepare the input for the model.
        inputs = [{"text": f"You are a helpful assistant. {task_instruction}"}]

        if dataset in DATASETS or "conditioning" in dataset:
            for i in range(len(n_shot_support)):
                image_path = n_shot_support[i]["image"]
                inputs.append({"image": os.path.join(data_path, image_path)})

                inputs.append(
                    {
                        "text": "Assistant: "
                        + format_answer(n_shot_support[i]["answer"], dataset, query)
                        + "\n"
                    }
                )

            for query_image_path in query_image_paths:
                inputs.append({"image": query_image_path})
            inputs.append({"text": "Assistant:"})

        else:
            for i in range(len(n_shot_support)):
                for image_path in n_shot_support[i]["image"]:
                    inputs.append({"image": os.path.join(data_path, image_path)})

                inputs.append(
                    {
                        "text": "User: "
                        + n_shot_support[i]["question"]
                        + "\nAssistant: "
                        + format_answer(n_shot_support[i]["answer"], dataset, query)
                        + "\n"
                    }
                )

            for query_image_path in query_image_paths:
                inputs.append({"image": query_image_path})
            inputs.append({"text": "User: " + query_text + "\nAssistant: "})

        total_inputs = tokenizer.from_list_format(inputs)
        inputs = tokenizer(total_inputs, return_tensors="pt")
        inputs = inputs.to(model.device)

        # Generate the answer and the scores.
        with torch.no_grad():
            predicted_answers = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
            )

        input_token_len = inputs["input_ids"].shape[1]

        # Decode the model's answer too.
        predicted_answers = tokenizer.decode(
            predicted_answers["sequences"][:, input_token_len:].cpu()[0],
            skip_special_tokens=True,
        )

    elif "mmicl" in engine:
        image_palceholder = "图"
        sp = [image_palceholder] + [f"<image{i}>" for i in range(20)]
        sp = sp + processor.tokenizer.additional_special_tokens[len(sp) :]
        processor.tokenizer.add_special_tokens({"additional_special_tokens": sp})

        if model.qformer.embeddings.word_embeddings.weight.shape[0] != len(
            processor.qformer_tokenizer
        ):
            model.qformer.resize_token_embeddings(len(processor.qformer_tokenizer))
        replace_token = "".join(32 * [image_palceholder])

        images = []
        input_text = f"{task_instruction}\n"
        image_count = 0
        if dataset in DATASETS or "conditioning" in dataset:
            for i in range(len(n_shot_support)):
                image_path = n_shot_support[i]["image"]
                images.append(
                    Image.open(os.path.join(data_path, image_path)).convert("RGB")
                )
                input_text += f"image {image_count}: <image{i}>{replace_token} "
                image_count += 1
                input_text += f"Answer: {format_answer(n_shot_support[i]['answer'], dataset, query)}\n"
            for query_image in query_images:
                images.append(query_image)
                input_text += (
                    f"image {image_count}: <image{image_count}>{replace_token} "
                )
                image_count += 1

            input_text += f"Answer: "
        else:
            for i in range(len(n_shot_support)):
                for j, image_path in enumerate(n_shot_support[i]["image"]):
                    images.append(
                        Image.open(os.path.join(data_path, image_path)).convert("RGB")
                    )
                    input_text += f"image {image_count}: <image{j}>{replace_token}"
                    image_count += 1
                input_text += f"{n_shot_support[i]['question']}\nAnswer: {format_answer(n_shot_support[i]['answer'], dataset, query)}\n"
            for query_image in query_images:
                images.append(query_image)
                input_text += (
                    f"image {image_count}: <image{image_count}>{replace_token} "
                )
                image_count += 1

            input_text += f"{query_text}\nAnswer: "

        inputs = processor(images=images, text=input_text, return_tensors="pt")

        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        inputs["img_mask"] = torch.tensor([[1 for i in range(len(images))]])
        inputs["pixel_values"] = inputs["pixel_values"].unsqueeze(0)

        inputs = inputs.to("cuda:0")
        with torch.no_grad():
            predicted_answers = model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                img_mask=inputs["img_mask"],
                do_sample=False,
                max_new_tokens=max_new_tokens,
                min_length=1,
                set_min_padding_size=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Decode the model's answer too.
        predicted_answers = tokenizer.decode(
            predicted_answers["sequences"][:, :].cpu()[0],
            skip_special_tokens=True,
        )

    elif "llava" in engine:
        images = []
        input_text = f"{task_instruction}\n"
        if dataset in DATASETS or "conditioning" in dataset:
            for i in range(len(n_shot_support)):
                image_path = n_shot_support[i]["image"]
                images.append(
                    Image.open(os.path.join(data_path, image_path)).convert("RGB")
                )
                input_text += f"{DEFAULT_IMAGE_TOKEN}\n"
                input_text += f"Answer: {format_answer(n_shot_support[i]['answer'], dataset, query)}\n"

            for query_image in query_images:
                images.append(query_image)
                input_text += f"{DEFAULT_IMAGE_TOKEN}\n"
            input_text += f"Answer: "
        else:
            for i in range(len(n_shot_support)):
                for image_path in n_shot_support[i]["image"]:
                    images.append(
                        Image.open(os.path.join(data_path, image_path)).convert("RGB")
                    )
                    input_text += f"{DEFAULT_IMAGE_TOKEN}\n"
                input_text += f"{n_shot_support[i]['question']}\nAnswer: {format_answer(n_shot_support[i]['answer'], dataset, query)}\n"

            for query_image in query_images:
                images.append(query_image)
                input_text += f"{DEFAULT_IMAGE_TOKEN}\n"
            input_text += f"{query_text}\nAnswer:"
        image_tensor = torch.stack(
            [
                processor.preprocess(image_file, return_tensors="pt")["pixel_values"][0]
                for image_file in images
            ]
        )

        image_tensor = image_tensor.half().cuda()
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], input_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            predicted_answers = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
            )
        input_token_len = input_ids.shape[1]

        # Decode the model's answer too.
        predicted_answers = tokenizer.batch_decode(
            predicted_answers["sequences"][:, :],
            skip_special_tokens=True,
        )[0]

    elif "flamingo" in engine:
        images = []
        input_text = f"{task_instruction}\n"
        if dataset in DATASETS or "conditioning" in dataset:
            for i in range(len(n_shot_support)):
                for image_path in n_shot_support[i]["image"]:
                    images.append(
                        Image.open(os.path.join(data_path, image_path)).convert("RGB")
                    )
                    input_text += "<image>"
                input_text += f"Answer: {format_answer(n_shot_support[i]['answer'], dataset, query)}\n"
            for query_image in query_images:
                images.append(query_image)
                input_text += "<image>"
        else:
            for i in range(len(n_shot_support)):
                for image_path in n_shot_support[i]["image"]:
                    images.append(
                        Image.open(os.path.join(data_path, image_path)).convert("RGB")
                    )
                    input_text += "<image>"
                input_text += f"{n_shot_support[i]['question']}\nAnswer: {format_answer(n_shot_support[i]['answer'], dataset, query)}\n"
            for query_image in query_images:
                images.append(query_image)
                input_text += "<image>"

        vision_x = [processor(image).unsqueeze(0) for image in images]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)

        if dataset in DATASETS or "conditioning" in dataset:
            input_text += f"Answer:"
        else:
            input_text += f"{query_text}\nAnswer:"

        lang_x = tokenizer(
            [input_text],
            return_tensors="pt",
        )
        with torch.no_grad():
            predicted_answers = model.generate(
                vision_x=vision_x.to(torch.bfloat16).cuda(),
                lang_x=lang_x["input_ids"].cuda(),
                attention_mask=lang_x["attention_mask"].cuda(),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        input_token_len = lang_x["input_ids"].shape[1]

        # Decode the model's answer too.
        predicted_answers = tokenizer.decode(
            predicted_answers["sequences"][:, input_token_len:].cpu()[0],
            skip_special_tokens=True,
        )

    elif "otter" in engine:
        images = []
        input_text = f"{task_instruction}\n"
        if dataset in DATASETS or "conditioning" in dataset:
            for i in range(len(n_shot_support)):
                image_path = n_shot_support[i]["image"]
                images.append(
                    Image.open(os.path.join(data_path, image_path)).convert("RGB")
                )
                input_text += "<image>"
                input_text += f"GPT:<answer> {format_answer(n_shot_support[i]['answer'], dataset, query)}<|endofchunk|>"
            for query_image in query_images:
                images.append(query_image)
                input_text += "<image>"
            input_text += f"GPT:<answer>"
        else:
            for i in range(len(n_shot_support)):
                for image_path in n_shot_support[i]["image"]:
                    images.append(
                        Image.open(os.path.join(data_path, image_path)).convert("RGB")
                    )
                    input_text += "<image>"
                input_text += f"User: {n_shot_support[i]['question']}\nGPT:<answer> {format_answer(n_shot_support[i]['answer'], dataset, query)}<|endofchunk|>"
            for query_image in query_images:
                images.append(query_image)
                input_text += "<image>"
            input_text += f"User: {query_text}\nGPT:<answer> "

        vision_x = (
            processor.preprocess(images, return_tensors="pt")["pixel_values"]
            .unsqueeze(1)
            .unsqueeze(0)
        )

        lang_x = model.text_tokenizer(
            [
                input_text,
            ],
            return_tensors="pt",
        )
        bad_words_id = tokenizer(
            ["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False
        ).input_ids

        # Generate the answer and the scores.
        with torch.no_grad():
            predicted_answers = model.generate(
                vision_x=vision_x.to(model.device),
                lang_x=lang_x["input_ids"].to(model.device),
                attention_mask=lang_x["attention_mask"].to(model.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                bad_words_ids=bad_words_id,
                output_scores=True,
                return_dict_in_generate=True,
            )

        input_token_len = lang_x["input_ids"].shape[1]

        predicted_answers = tokenizer.decode(
            predicted_answers["sequences"][:, input_token_len:].cpu()[0],
            skip_special_tokens=True,
        )

    elif "emu2-chat" in engine:
        images = []
        input_text = f"{task_instruction}\n"
        if dataset in DATASETS or "conditioning" in dataset:
            for i in range(len(n_shot_support)):
                for image_path in n_shot_support[i]["image"]:
                    images.append(
                        Image.open(os.path.join(data_path, image_path)).convert("RGB")
                    )
                    input_text += "[<IMG_PLH>]"
                input_text += f"[Answer: {format_answer(n_shot_support[i]['answer'], dataset, query)}]."
            for query_image in query_images:
                images.append(query_image)
                input_text += "[<IMG_PLH>]"
            input_text += f"Answer:"
        else:
            for i in range(len(n_shot_support)):
                for image_path in n_shot_support[i]["image"]:
                    images.append(
                        Image.open(os.path.join(data_path, image_path)).convert("RGB")
                    )
                    input_text += "[<IMG_PLH>]"
                input_text += f"[{n_shot_support[i]['question']}\nAnswer: {format_answer(n_shot_support[i]['answer'], dataset, query)}]."
            for query_image in query_images:
                images.append(query_image)
                input_text += "[<IMG_PLH>]"
            input_text += f"[{query_text}\nAnswer:"
        inputs = model.build_input_ids(
            text=[input_text], tokenizer=tokenizer, image=images
        )

        with torch.no_grad():
            predicted_answers = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image=inputs["image"].to(torch.bfloat16),
                max_new_tokens=max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
            )
        input_token_len = inputs["input_ids"].shape[1]

        # Decode the model's answer too.
        predicted_answers = tokenizer.decode(
            predicted_answers["sequences"][:, :].cpu()[0],
            skip_special_tokens=True,
        )

    elif "idefics" in engine:
        prompts = [f"You are a helpful assistant.\n{task_instruction}\n"]
        for i in range(len(n_shot_support)):
            image_path = n_shot_support[i]["image"]
            prompts.append(
                Image.open(os.path.join(data_path, image_path)).convert("RGB")
            )
            if dataset in DATASETS or "conditioning" in dataset:
                prompts.append(
                    f"\nAssistant: {format_answer(n_shot_support[i]['answer'], dataset, query)}\n"
                )
            else:
                prompts.append(f"\nUser: {n_shot_support[i]['question']}")
                # prompts.append("<end_of_utterance>")
                prompts.append(
                    f"\nAssistant: {format_answer(n_shot_support[i]['answer'], dataset, query)}\n"
                )

        if dataset in DATASETS or "conditioning" in dataset:
            for query_image in query_images:
                prompts.append(query_image)
                # Convert query image to tensor

            prompts.append("\nAssistant: ")
        else:
            for query_image in query_images:
                prompts.append(query_image)
            prompts.append(f"\nUser: {query_text}")
            # prompts.append("<end_of_utterance>")
            prompts.append("\nAssistant: ")
        inputs = processor(
            prompts, add_end_of_utterance_token=False, return_tensors="pt"
        ).to("cuda")

        exit_condition = processor.tokenizer(
            "<end_of_utterance>", add_special_tokens=False
        ).input_ids
        bad_words_ids = processor.tokenizer(
            ["<image>", "<fake_token_around_image>"], add_special_tokens=False
        ).input_ids

        predicted_answers = model.generate(
            **inputs,
            eos_token_id=exit_condition,
            bad_words_ids=bad_words_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
        input_token_len = inputs["input_ids"].shape[1]

        # Decode the model's answer too.
        predicted_answers = tokenizer.decode(
            predicted_answers["sequences"][:, input_token_len:].cpu()[0],
            skip_special_tokens=True,
        )

    return predicted_answers
