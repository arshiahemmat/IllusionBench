import os
import torch
from PIL import Image
from typing import Any, Dict, List, Optional

from .ICL_utils import get_task_instruction, format_answer
from .utils import load_image
from .constants import ALL_TASKS, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX

try:
    from llava.conversation import conv_templates
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.mm_utils import tokenizer_image_token
except ImportError:
    pass


@torch.no_grad()
def icl_inference(
    args: Any,
    engine: str,
    dataset: str,
    model: torch.nn.Module,
    tokenizer: Any,
    query: Dict[str, Any],
    n_shot_support: List[Dict[str, Any]],
    data_path: str,
    processor: Any,
    max_new_tokens: int,
) -> str:
    """
    Main inference function that delegates to engine-specific helper functions.

    Args:
        args (Any): Arguments for task instructions.
        engine (str): The engine to use for inference.
        dataset (str): The dataset being used.
        model (torch.nn.Module): The model to use for inference.
        tokenizer (Any): Tokenizer for processing text.
        query (Dict[str, Any]): The query containing the image and question.
        n_shot_support (List[Dict[str, Any]]): Support examples for few-shot learning.
        data_path (str): Path to the data directory.
        processor (Any): Processor for handling images and text.
        max_new_tokens (int): Maximum number of new tokens to generate.

    Returns:
        str: The generated answer from the model.
    """
    task_instruction = get_task_instruction(dataset, args)
    img_id = query.get("image")
    query_images, query_image_paths = load_image(img_id, data_path)

    # Retrieve the query text if applicable
    query_text = query.get("question") if (dataset not in ALL_TASKS and "conditioning" not in dataset) else None

    # Delegate to the appropriate inference function based on the engine
    if "qwen-vl" in engine:
        predicted_answers = _inference_qwen_vl(
            task_instruction,
            dataset,
            n_shot_support,
            data_path,
            query,
            query_images,
            query_image_paths,
            query_text,
            model,
            tokenizer,
            processor,
            max_new_tokens,
        )
    elif "mmicl" in engine:
        predicted_answers = _inference_mmicl(
            task_instruction,
            dataset,
            n_shot_support,
            data_path,
            query,
            query_images,
            query_image_paths,
            query_text,
            model,
            tokenizer,
            processor,
            max_new_tokens,
        )
    elif "llava" in engine:
        predicted_answers = _inference_llava(
            task_instruction,
            dataset,
            n_shot_support,
            data_path,
            query,
            query_images,
            query_image_paths,
            query_text,
            model,
            tokenizer,
            processor,
            max_new_tokens,
        )
    elif "otter" in engine:
        predicted_answers = _inference_otter(
            task_instruction,
            dataset,
            n_shot_support,
            data_path,
            query,
            query_images,
            query_image_paths,
            query_text,
            model,
            tokenizer,
            processor,
            max_new_tokens,
        )
    elif "idefics" in engine:
        predicted_answers = _inference_idefics(
            task_instruction,
            dataset,
            n_shot_support,
            data_path,
            query,
            query_images,
            query_image_paths,
            query_text,
            model,
            tokenizer,
            processor,
            max_new_tokens,
        )
    else:
        raise ValueError(f"Unsupported engine: {engine}")

    return predicted_answers


def _inference_qwen_vl(
    task_instruction: str,
    dataset: str,
    n_shot_support: List[Dict[str, Any]],
    data_path: str,
    query: Dict[str, Any],
    query_images: List[Image.Image],
    query_image_paths: List[str],
    query_text: Optional[str],
    model: torch.nn.Module,
    tokenizer: Any,
    processor: Any,
    max_new_tokens: int,
) -> str:
    """
    Performs inference using the Qwen-VL engine.

    Args:
        task_instruction (str): Instruction for the task based on the dataset.
        dataset (str): The dataset being used.
        n_shot_support (List[Dict[str, Any]]): Support examples for few-shot learning.
        data_path (str): Path to the data directory.
        query (Dict[str, Any]): The query containing the image and question.
        query_images (List[Image.Image]): Loaded query images.
        query_image_paths (List[str]): Paths to the query images.
        query_text (Optional[str]): The question text if applicable.
        model (torch.nn.Module): The model to use for inference.
        tokenizer (Any): Tokenizer for processing text.
        processor (Any): Processor for handling images and text.
        max_new_tokens (int): Maximum number of new tokens to generate.

    Returns:
        str: The generated answer from the model.
    """
    inputs = [{"text": f"You are a helpful assistant. {task_instruction}"}]

    if dataset in ALL_TASKS or "conditioning" in dataset:
        # Few-shot examples with dataset tasks or conditioning
        for support in n_shot_support:
            image_path = support["image"]
            inputs.append({"image": os.path.join(data_path, image_path)})
            answer_text = f"Assistant: {format_answer(support['answer'], dataset, query)}\n"
            inputs.append({"text": answer_text})
        for img_path in query_image_paths:
            inputs.append({"image": img_path})
        inputs.append({"text": "Assistant:"})
    else:
        # Few-shot examples without dataset tasks or conditioning
        for support in n_shot_support:
            for image_path in support["image"]:
                inputs.append({"image": os.path.join(data_path, image_path)})
            user_text = f"User: {support['question']}\nAssistant: {format_answer(support['answer'], dataset, query)}\n"
            inputs.append({"text": user_text})
        for img_path in query_image_paths:
            inputs.append({"image": img_path})
        user_query_text = f"User: {query_text}\nAssistant: "
        inputs.append({"text": user_query_text})

    # Tokenize inputs
    total_inputs = tokenizer.from_list_format(inputs)
    inputs_tensor = tokenizer(total_inputs, return_tensors="pt").to(model.device)

    # Generate outputs
    outputs = model.generate(
        **inputs_tensor,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        min_new_tokens=1,
        output_scores=True,
        return_dict_in_generate=True,
    )

    # Decode the generated answer
    input_length = inputs_tensor["input_ids"].shape[1]
    answer = tokenizer.decode(
        outputs["sequences"][:, input_length:].cpu()[0],
        skip_special_tokens=True,
    )

    return answer


def _inference_mmicl(
    task_instruction: str,
    dataset: str,
    n_shot_support: List[Dict[str, Any]],
    data_path: str,
    query: Dict[str, Any],
    query_images: List[Image.Image],
    query_image_paths: List[str],
    query_text: Optional[str],
    model: torch.nn.Module,
    tokenizer: Any,
    processor: Any,
    max_new_tokens: int,
) -> str:
    """
    Performs inference using the MMICL engine.

    Args:
        task_instruction (str): Instruction for the task based on the dataset.
        dataset (str): The dataset being used.
        n_shot_support (List[Dict[str, Any]]): Support examples for few-shot learning.
        data_path (str): Path to the data directory.
        query (Dict[str, Any]): The query containing the image and question.
        query_images (List[Image.Image]): Loaded query images.
        query_image_paths (List[str]): Paths to the query images.
        query_text (Optional[str]): The question text if applicable.
        model (torch.nn.Module): The model to use for inference.
        tokenizer (Any): Tokenizer for processing text.
        processor (Any): Processor for handling images and text.
        max_new_tokens (int): Maximum number of new tokens to generate.

    Returns:
        str: The generated answer from the model.
    """
    image_placeholder = "图"
    special_tokens = [image_placeholder] + [f"<image{i}>" for i in range(20)]
    special_tokens += processor.tokenizer.additional_special_tokens[len(special_tokens):]
    processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # Resize token embeddings if necessary
    if model.qformer.embeddings.word_embeddings.weight.shape[0] != len(processor.qformer_tokenizer):
        model.qformer.resize_token_embeddings(len(processor.qformer_tokenizer))

    replace_token = " " * 32  # Placeholder replacement

    images = []
    input_text = f"{task_instruction}\n"
    image_count = 0

    if dataset in ALL_TASKS or "conditioning" in dataset:
        # Few-shot examples with dataset tasks or conditioning
        for support in n_shot_support:
            image_path = support["image"]
            images.append(Image.open(os.path.join(data_path, image_path)).convert("RGB"))
            input_text += f"image {image_count}: <image{image_count}>{replace_token} "
            image_count += 1
            input_text += f"Answer: {format_answer(support['answer'], dataset, query)}\n"
        for img in query_images:
            images.append(img)
            input_text += f"image {image_count}: <image{image_count}>{replace_token} "
            image_count += 1
        input_text += "Answer: "
    else:
        # Few-shot examples without dataset tasks or conditioning
        for support in n_shot_support:
            for j, image_path in enumerate(support["image"]):
                images.append(Image.open(os.path.join(data_path, image_path)).convert("RGB"))
                input_text += f"image {image_count}: <image{j}>{replace_token}"
                image_count += 1
            input_text += f"{support['question']}\nAnswer: {format_answer(support['answer'], dataset, query)}\n"
        for img in query_images:
            images.append(img)
            input_text += f"image {image_count}: <image{image_count}>{replace_token} "
            image_count += 1
        input_text += f"{query_text}\nAnswer: "

    # Process inputs
    inputs = processor(images=images, text=input_text, return_tensors="pt")
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16).unsqueeze(0)
    inputs["img_mask"] = torch.ones((1, len(images)), dtype=torch.long)

    # Move inputs to GPU
    inputs = inputs.to("cuda:0")

    # Generate outputs
    outputs = model.generate(
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

    # Decode the generated answer
    answer = tokenizer.decode(
        outputs["sequences"][:, :].cpu()[0],
        skip_special_tokens=True,
    )

    return answer


def _inference_llava(
    task_instruction: str,
    dataset: str,
    n_shot_support: List[Dict[str, Any]],
    data_path: str,
    query: Dict[str, Any],
    query_images: List[Image.Image],
    query_image_paths: List[str],
    query_text: Optional[str],
    model: torch.nn.Module,
    tokenizer: Any,
    processor: Any,
    max_new_tokens: int,
) -> str:
    """
    Performs inference using the LLaVA engine.

    Args:
        task_instruction (str): Instruction for the task based on the dataset.
        dataset (str): The dataset being used.
        n_shot_support (List[Dict[str, Any]]): Support examples for few-shot learning.
        data_path (str): Path to the data directory.
        query (Dict[str, Any]): The query containing the image and question.
        query_images (List[Image.Image]): Loaded query images.
        query_image_paths (List[str]): Paths to the query images.
        query_text (Optional[str]): The question text if applicable.
        model (torch.nn.Module): The model to use for inference.
        tokenizer (Any): Tokenizer for processing text.
        processor (Any): Processor for handling images and text.
        max_new_tokens (int): Maximum number of new tokens to generate.

    Returns:
        str: The generated answer from the model.
    """
    images = []
    input_text = f"{task_instruction}\n"

    if dataset in ALL_TASKS or "conditioning" in dataset:
        # Few-shot examples with dataset tasks or conditioning
        for support in n_shot_support:
            image_path = support["image"]
            images.append(Image.open(os.path.join(data_path, image_path)).convert("RGB"))
            input_text += f"{DEFAULT_IMAGE_TOKEN}\n"
            input_text += f"Answer: {format_answer(support['answer'], dataset, query)}\n"
        for img in query_images:
            images.append(img)
            input_text += f"{DEFAULT_IMAGE_TOKEN}\n"
        input_text += "Answer: "
    else:
        # Few-shot examples without dataset tasks or conditioning
        for support in n_shot_support:
            for image_path in support["image"]:
                images.append(Image.open(os.path.join(data_path, image_path)).convert("RGB"))
                input_text += f"{DEFAULT_IMAGE_TOKEN}\n"
            input_text += f"{support['question']}\nAnswer: {format_answer(support['answer'], dataset, query)}\n"
        for img in query_images:
            images.append(img)
            input_text += f"{DEFAULT_IMAGE_TOKEN}\n"
        input_text += f"{query_text}\nAnswer:"

    # Preprocess images and prepare tensors
    image_tensor = torch.stack(
        [processor.preprocess(img, return_tensors="pt")["pixel_values"][0] for img in images]
    ).half().cuda()

    # Prepare conversation template
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], input_text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize prompt with image tokens
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    # Generate outputs
    outputs = model.generate(
        input_ids,
        images=image_tensor,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        min_new_tokens=1,
        return_dict_in_generate=True,
        output_scores=True,
    )

    # Decode the generated answer
    decoded_answers = tokenizer.batch_decode(
        outputs["sequences"][:, :].cpu(),
        skip_special_tokens=True,
    )[0]

    return decoded_answers


def _inference_otter(
    task_instruction: str,
    dataset: str,
    n_shot_support: List[Dict[str, Any]],
    data_path: str,
    query: Dict[str, Any],
    query_images: List[Image.Image],
    query_image_paths: List[str],
    query_text: Optional[str],
    model: torch.nn.Module,
    tokenizer: Any,
    processor: Any,
    max_new_tokens: int,
) -> str:
    """
    Performs inference using the Otter engine.

    Args:
        task_instruction (str): Instruction for the task based on the dataset.
        dataset (str): The dataset being used.
        n_shot_support (List[Dict[str, Any]]): Support examples for few-shot learning.
        data_path (str): Path to the data directory.
        query (Dict[str, Any]): The query containing the image and question.
        query_images (List[Image.Image]): Loaded query images.
        query_image_paths (List[str]): Paths to the query images.
        query_text (Optional[str]): The question text if applicable.
        model (torch.nn.Module): The model to use for inference.
        tokenizer (Any): Tokenizer for processing text.
        processor (Any): Processor for handling images and text.
        max_new_tokens (int): Maximum number of new tokens to generate.

    Returns:
        str: The generated answer from the model.
    """
    images = []
    input_text = f"{task_instruction}\n"

    if dataset in ALL_TASKS or "conditioning" in dataset:
        # Few-shot examples with dataset tasks or conditioning
        for support in n_shot_support:
            image_path = support["image"]
            images.append(Image.open(os.path.join(data_path, image_path)).convert("RGB"))
            input_text += "<image>GPT:<answer> "
            input_text += f"{format_answer(support['answer'], dataset, query)}<|endofchunk|>"
        for img in query_images:
            images.append(img)
            input_text += "<image>"
        input_text += "GPT:<answer> "
    else:
        # Few-shot examples without dataset tasks or conditioning
        for support in n_shot_support:
            for image_path in support["image"]:
                images.append(Image.open(os.path.join(data_path, image_path)).convert("RGB"))
                input_text += "<image>"
            input_text += f"User: {support['question']}\nGPT:<answer> {format_answer(support['answer'], dataset, query)}<|endofchunk|>"
        for img in query_images:
            images.append(img)
            input_text += "<image>"
        input_text += f"User: {query_text}\nGPT:<answer> "

    # Preprocess images and prepare tensors
    vision_x = processor.preprocess(images, return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0).to(model.device)
    lang_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

    # Define bad words to avoid in generation
    bad_words = tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
    bad_words_ids = [ids for sublist in bad_words for ids in sublist]

    # Generate outputs
    outputs = model.generate(
        vision_x=vision_x,
        lang_x=lang_inputs["input_ids"],
        attention_mask=lang_inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        bad_words_ids=bad_words_ids,
        output_scores=True,
        return_dict_in_generate=True,
    )

    # Decode the generated answer
    input_length = lang_inputs["input_ids"].shape[1]
    answer = tokenizer.decode(
        outputs["sequences"][:, input_length:].cpu()[0],
        skip_special_tokens=True,
    )

    return answer


def _inference_idefics(
    task_instruction: str,
    dataset: str,
    n_shot_support: List[Dict[str, Any]],
    data_path: str,
    query: Dict[str, Any],
    query_images: List[Image.Image],
    query_image_paths: List[str],
    query_text: Optional[str],
    model: torch.nn.Module,
    tokenizer: Any,
    processor: Any,
    max_new_tokens: int,
) -> str:
    """
    Performs inference using the IDefics engine.

    Args:
        task_instruction (str): Instruction for the task based on the dataset.
        dataset (str): The dataset being used.
        n_shot_support (List[Dict[str, Any]]): Support examples for few-shot learning.
        data_path (str): Path to the data directory.
        query (Dict[str, Any]): The query containing the image and question.
        query_images (List[Image.Image]): Loaded query images.
        query_image_paths (List[str]): Paths to the query images.
        query_text (Optional[str]): The question text if applicable.
        model (torch.nn.Module): The model to use for inference.
        tokenizer (Any): Tokenizer for processing text.
        processor (Any): Processor for handling images and text.
        max_new_tokens (int): Maximum number of new tokens to generate.

    Returns:
        str: The generated answer from the model.
    """
    prompts = [f"You are a helpful assistant.\n{task_instruction}\n"]
    images = []
    input_text = ""

    # Add few-shot support examples
    for support in n_shot_support:
        image_path = support["image"]
        images.append(Image.open(os.path.join(data_path, image_path)).convert("RGB"))
        prompts.append(images[-1])
        if dataset in ALL_TASKS or "conditioning" in dataset:
            prompts.append(f"\nAssistant: {format_answer(support['answer'], dataset, query)}\n")
        else:
            prompts.append(f"\nUser: {support['question']}")
            prompts.append(f"\nAssistant: {format_answer(support['answer'], dataset, query)}\n")

    if dataset in ALL_TASKS or "conditioning" in dataset:
        # Add query images and prompt for answer
        for img in query_images:
            images.append(img)
            prompts.append(img)
        prompts.append("\nAssistant: ")
    else:
        # Add query images and user query
        for img in query_images:
            images.append(img)
            prompts.append(img)
        prompts.append(f"\nUser: {query_text}")
        prompts.append("\nAssistant: ")

    # Process inputs
    inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to("cuda")

    # Define termination and bad words
    exit_ids = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
    bad_words_ids = [ids for sublist in bad_words for ids in sublist]

    # Generate outputs
    outputs = model.generate(
        **inputs,
        eos_token_id=exit_ids,
        bad_words_ids=bad_words_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
    )

    # Decode the generated answer
    input_length = inputs["input_ids"].shape[1]
    answer = tokenizer.decode(
        outputs["sequences"][:, input_length:].cpu()[0],
        skip_special_tokens=True,
    )

    return answer
