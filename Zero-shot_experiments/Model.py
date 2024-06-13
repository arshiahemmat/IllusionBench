from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForCausalLM, LlamaTokenizer


import torch
from PIL import Image
import requests





def find_model(model_name):
    if model_name == 'blipv2-t5':
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xxl", torch_dtype=torch.float16)
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
        template = 'Question: [ORGINAL] Answer: '

    if model_name == 'blipv2-7b':
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        template = 'Question: [ORGINAL] Answer: '


    if model_name == 'instructblip-7b':
        model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b",  torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
        template = '[ORGINAL]'



    if model_name == 'instructblip-13b':
        model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-13b",  torch_dtype=torch.float16)
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-13b")
        template = '[ORGINAL]'



    if model_name == 'instructblip-t5':
        model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xxl",  torch_dtype=torch.float16)
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xxl")
        template = '[ORGINAL]'



    if model_name == 'llava1.5-7b':
        model = LlavaForConditionalGeneration.from_pretrained( "llava-hf/llava-1.5-7b-hf",torch_dtype=torch.float16, low_cpu_mem_usage=True)
        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        template = 'USER: <image>\n[ORGINAL]\nASSISTANT:'

    if model_name == 'llava1.5-13b':
        model = LlavaForConditionalGeneration.from_pretrained( "llava-hf/llava-1.5-13b-hf",torch_dtype=torch.float16, low_cpu_mem_usage=True)
        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-13b-hf")
        template = 'USER: <image>\n[ORGINAL]\nASSISTANT:'


    if model_name == 'llava1.6-34b':
        model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-34b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
        processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-34b-hf")
        template ='[INST] <image>\n[ORGINAL]\n[/INST]'


    if model_name == 'llava1.6-13b':
        processor = AutoProcessor.from_pretrained("liuhaotian/llava-v1.6-vicuna-13b")
        model = AutoModelForCausalLM.from_pretrained("liuhaotian/llava-v1.6-vicuna-13b")

        template ='[INST] <image>\n[ORGINAL]\n[/INST]'

    if model_name == 'llava1.6-7b':
        model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
        processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        template ='[INST] <image>\n[ORGINAL]\n[/INST]'


    if model_name == 'cogvlm':
        processor = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
        model = AutoModelForCausalLM.from_pretrained('THUDM/cogvlm-chat-hf',torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,trust_remote_code=True) # EVAL NEED

    return processor, model,template