import transformers
import torch


def load_i2t_model(engine, args=None):
    """Load the specified I2T model based on the engine type and return the model, tokenizer, and processor."""
    
    device = args.engine_device if args else "cuda:0"
    dtype = torch.bfloat16  # Set default torch dtype for all models

    if engine == "otter-mpt":
        return _load_otter_mpt_model(device, dtype)
    
    elif engine == "mmicl-t5-xxl":
        return _load_mmicl_t5_xxl_model(device, dtype)
    
    elif engine == "llava16-7b":
        return _load_llava16_7b_model(device, dtype, args)
    
    elif engine == "qwen-vl-chat":
        return _load_qwen_model("Qwen/Qwen-VL-Chat", device, dtype)
    
    elif engine == "qwen-vl":
        return _load_qwen_model("Qwen/Qwen-VL", device, dtype)
    
    elif engine == "idefics-9b-instruct":
        return _load_idefics_model(device, dtype)
    
    else:
        raise NotImplementedError(f"Engine {engine} not recognized.")


def _load_otter_mpt_model(device, dtype):
    """Load and return the Otter-MPT model, tokenizer, and processor."""
    from otter_ai import OtterForConditionalGeneration

    model = OtterForConditionalGeneration.from_pretrained(
        "luodian/OTTER-Image-MPT7B", torch_dtype=dtype
    ).to(device)
    tokenizer = model.text_tokenizer
    image_processor = transformers.CLIPImageProcessor()
    
    # Print the number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters())
    print(f"Number of trainable parameters: {trainable_params}")
    
    return model, tokenizer, image_processor


def _load_mmicl_t5_xxl_model(device, dtype):
    """Load and return the MMICL-T5-XXL model, tokenizer, and processor."""
    from instructblip import InstructBlipConfig, InstructBlipForConditionalGeneration, InstructBlipProcessor

    model_ckpt = "BleachNick/MMICL-Instructblip-T5-xxl"
    processor_ckpt = "Salesforce/instructblip-flan-t5-xxl"
    
    config = InstructBlipConfig.from_pretrained(model_ckpt)
    model = InstructBlipForConditionalGeneration.from_pretrained(
        model_ckpt, config=config
    ).to(device, dtype=dtype)
    
    processor = InstructBlipProcessor.from_pretrained(processor_ckpt)
    tokenizer = processor.tokenizer

    return model, tokenizer, processor


def _load_llava16_7b_model(device, dtype, args):
    """Load and return the LLAVA-16-7B model, tokenizer, and processor."""
    from llava.model.builder import load_pretrained_model as load_llava_model

    tokenizer, model, image_processor, context_len = load_llava_model(
        model_path="liuhaotian/llava-v1.6-vicuna-7b",
        model_base=None,
        device_map=args.engine_device,
        model_name="llava",
        torch_dtype=dtype,
    )
    
    model = model.to(device)
    
    return model, tokenizer, image_processor


def _load_qwen_model(model_name, device, dtype):
    """Load and return the Qwen-VL or Qwen-VL-Chat model, tokenizer, and generation configuration."""
    from transformers.generation import GenerationConfig

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).eval().to(device)

    model.generation_config = GenerationConfig.from_pretrained(model_name, trust_remote_code=True)

    return model, tokenizer, None  # No processor for Qwen models


def _load_idefics_model(device, dtype):
    """Load and return the IDEFICS-9B-INSTRUCT model, tokenizer, and processor."""
    from transformers import IdeficsForVisionText2Text, AutoProcessor

    checkpoint = "HuggingFaceM4/idefics-9b-instruct"
    model = IdeficsForVisionText2Text.from_pretrained(
        checkpoint,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    
    processor = AutoProcessor.from_pretrained(checkpoint)
    tokenizer = processor.tokenizer

    return model, tokenizer, processor
