import transformers
import torch


def load_i2t_model(engine, args=None):
    if engine == "otter-mpt":
        from otter_ai import OtterForConditionalGeneration

        model = OtterForConditionalGeneration.from_pretrained(
            "luodian/OTTER-Image-MPT7B", torch_dtype=torch.bfloat16
        ).to(args.engine_device)
        tokenizer = model.text_tokenizer
        image_processor = transformers.CLIPImageProcessor()
        processor = image_processor
        trainable_params = sum(p.numel() for p in model.parameters())
        print(f"Number of trainable parameters: {trainable_params}")
    elif engine == "mmicl-t5-xxl":
        from instructblip import (
            InstructBlipConfig,
            InstructBlipForConditionalGeneration,
            InstructBlipProcessor,
        )
        model_type = "instructblip"
        model_ckpt = "BleachNick/MMICL-Instructblip-T5-xxl"
        processor_ckpt = "Salesforce/instructblip-flan-t5-xxl"
        config = InstructBlipConfig.from_pretrained(model_ckpt)

        if "instructblip" in model_type:
            model = InstructBlipForConditionalGeneration.from_pretrained(
                model_ckpt, config=config
            ).to(args.engine_device, dtype=torch.bfloat16)

            # Print number of trainable parameters of the model
            

        processor = InstructBlipProcessor.from_pretrained(processor_ckpt)
        tokenizer = processor.tokenizer
    elif engine == "mmicl-t5-xl":
        from instructblip import (
            InstructBlipConfig,
            InstructBlipForConditionalGeneration,
            InstructBlipProcessor,
        )

        model_type = "instructblip"
        model_ckpt = "BleachNick/MMICL-Instructblip-T5-xl"
        processor_ckpt = "Salesforce/instructblip-flan-t5-xl"
        config = InstructBlipConfig.from_pretrained(model_ckpt)

        if "instructblip" in model_type:
            model = InstructBlipForConditionalGeneration.from_pretrained(
                model_ckpt, config=config
            ).to(args.engine_device, dtype=torch.bfloat16)


        processor = InstructBlipProcessor.from_pretrained(processor_ckpt)
        tokenizer = processor.tokenizer

    elif engine == "otter-llama":
        from otter_ai import OtterForConditionalGeneration

        model = OtterForConditionalGeneration.from_pretrained(
            "luodian/OTTER-9B-LA-InContext",
            torch_dtype=torch.bfloat16,
        ).to(args.engine_device)
        tokenizer = model.text_tokenizer
        image_processor = transformers.CLIPImageProcessor()
        processor = image_processor
    elif engine == "llava16-7b":
        from llava.model.builder import load_pretrained_model as load_llava_model

        tokenizer, model, image_processor, context_len = load_llava_model(
            model_path="liuhaotian/llava-v1.6-vicuna-7b",
            model_base=None,
            device_map=args.engine_device,
            model_name="llava",
            torch_dtype=torch.bfloat16,
        )
        model = model.to(args.engine_device)
        processor = image_processor

    elif engine == "qwen-vl-chat":
        from transformers.generation import GenerationConfig

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen-VL-Chat", trust_remote_code=True
        )
        model = (
            transformers.AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen-VL-Chat",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            .eval()
            .to(args.engine_device)
        )
        model.generation_config = GenerationConfig.from_pretrained(
            "Qwen/Qwen-VL-Chat", trust_remote_code=True
        )
        processor = None

    elif engine == "qwen-vl":
        from transformers.generation import GenerationConfig

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen-VL", trust_remote_code=True
        )
        model = (
            transformers.AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen-VL",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            .eval()
            .to(args.engine_device)
        )
        model.generation_config = GenerationConfig.from_pretrained(
            "Qwen/Qwen-VL", trust_remote_code=True
        )
        processor = None
    elif engine == "internlm-x2":
        model = transformers.AutoModel.from_pretrained(
            "internlm/internlm-xcomposer2-7b",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(args.engine_device)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "internlm/internlm-xcomposer2-7b", trust_remote_code=True
        )
        model.tokenizer = tokenizer
        processor = None
    elif engine == "openflamingo":
        from open_flamingo import create_model_and_transforms

        model, processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-7b",
            tokenizer_path="anas-awadalla/mpt-7b",
            cross_attn_every_n_layers=4,
        )
        model = model.to(torch.bfloat16).to(args.engine_device)

    elif engine == "emu2-chat":
        from accelerate import (
            init_empty_weights,
            infer_auto_device_map,
            load_checkpoint_and_dispatch,
        )

        tokenizer = transformers.AutoTokenizer.from_pretrained("BAAI/Emu2-Chat")
        with init_empty_weights():
            model = transformers.AutoModelForCausalLM.from_pretrained(
                "BAAI/Emu2-Chat",
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).eval()

        # adjust according to your device
        device_map = infer_auto_device_map(
            model,
            max_memory={0: "38GiB", 1: "38GiB", 2: "38GiB", 3: "38GiB"},
            no_split_module_classes=["Block", "LlamaDecoderLayer"],
        )
        device_map["model.decoder.lm.lm_head"] = 0

        model = load_checkpoint_and_dispatch(
            model,
            "/scratch/local/ssd/tomlamb/hub/models--BAAI--Emu2-Chat/snapshots/20ea30b04f8fee599cf97535e655c200df728501",
            device_map=device_map,
        ).eval()
        processor = None

    elif engine == "idefics-9b-instruct":
        from transformers import IdeficsForVisionText2Text, AutoProcessor

        checkpoint = "HuggingFaceM4/idefics-9b-instruct"
        model = IdeficsForVisionText2Text.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(args.engine_device)
        processor = AutoProcessor.from_pretrained(checkpoint)
        tokenizer = processor.tokenizer
    elif engine == "idefics-9b":
        from transformers import IdeficsForVisionText2Text, AutoProcessor

        checkpoint = "HuggingFaceM4/idefics-9b"
        model = IdeficsForVisionText2Text.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(args.engine_device)
        processor = AutoProcessor.from_pretrained(checkpoint)
        tokenizer = processor.tokenizer
    elif engine == "idefics-80b-instruct":
        from transformers import IdeficsForVisionText2Text, AutoProcessor
        from accelerate import (
            init_empty_weights,
            infer_auto_device_map,
            load_checkpoint_and_dispatch,
        )

        checkpoint = "HuggingFaceM4/idefics-80b-instruct"
        model = IdeficsForVisionText2Text.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(checkpoint)
        tokenizer = processor.tokenizer
    elif engine == "gpt4v":
        model, tokenizer, processor = None, None, None
    else:
        raise NotImplementedError
    return model, tokenizer, processor
