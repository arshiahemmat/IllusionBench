from transformers import AutoProcessor, LlavaForConditionalGeneration  
import torch
from PIL import Image
import requests





def llava_find_model(model_name):
    if model_name == 'llava1.5-7b':
        model = LlavaForConditionalGeneration.from_pretrained( "llava-hf/llava-1.5-7b-hf",torch_dtype=torch.float16, low_cpu_mem_usage=True)
        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        template = 'USER: <image>\n[ORGINAL]\nASSISTANT:'

    if model_name == 'llava1.5-13b':
        model = LlavaForConditionalGeneration.from_pretrained( "llava-hf/llava-1.5-13b-hf",torch_dtype=torch.float16, low_cpu_mem_usage=True)
        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-13b-hf")
        template = 'USER: <image>\n[ORGINAL]\nASSISTANT:'

    return processor, model,template