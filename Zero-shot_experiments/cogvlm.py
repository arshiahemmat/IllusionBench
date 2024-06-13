import torch
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
import argparse
from Model import find_model
import torch
from PIL import Image
from torch.nn.functional import softmax
import requests
import argparse
import os
import pandas as pd
import pickle
import random
import string
import re
import shutil


os.environ['TRANSFORMERS_CACHE'] = '/scratch/local/ssd/arshia/huggingface/hub/'
os.environ['HF_HOME'] = '/scratch/local/ssd/arshia/huggingface/hub/'





def find_folder_containing_string(root_path, search_string):
    for dirpath, dirnames, filenames in os.walk(root_path):
        for dirname in dirnames:
            if search_string in dirname:
                return os.path.join(dirpath, dirname)
    return None


def copy_file(source_file, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    shutil.copy2(source_file, destination_folder)


def remove_file(file_path):
    os.remove(file_path)



def model_response(tokenizer, model,image, query, device, num_beams=5, llava_mode=False):
    inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image], template_version='vqa')   # vqa mode
    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to(device),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(device),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to(device),
        'images': [[inputs['images'][0].to(device).to(torch.bfloat16)]],
    }
    gen_kwargs = {"max_length": 2048, "do_sample": False}

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        prediction = tokenizer.decode(outputs[0])

    prediction = prediction.replace("</s>", "")
    return prediction.lower()



def logprob_out_given_prompt(prompt, output, raw_image, model, processor):
    """ Calculate the log probability of output given a prompt for a multimodal model """
    # Assume `raw_image` is a preloaded image tensor or similar object compatible with the processor
    input_dict = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)
    output_dict = processor(output, raw_image, return_tensors='pt').to(0, torch.float16)

    # Concatenate input_ids for prompt and output
    input_ids = torch.cat([input_dict['input_ids'], output_dict['input_ids'][:, 1:]], dim=1)  # Avoid repeating the start token

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits

    log_sum = 0.0
    # Calculate log probability for each output token
    for i in range(input_dict['input_ids'].shape[1], input_ids.shape[1]):
        token_logit = logits[0, i]
        token_log_probs = torch.log_softmax(token_logit, dim=-1)
        log_token_prob = token_log_probs[input_ids[0, i]].item()
        log_sum += log_token_prob

    return log_sum

def prob_outs_given_prompt(prompt, outputs, raw_image, model, processor):
    """ Calculate probabilities for multiple outputs given a single prompt """
    logprobs = [logprob_out_given_prompt(prompt, out, raw_image, model, processor) for out in outputs]
    probs = softmax(torch.tensor(logprobs), dim=0)
    return {out: probs[i].item() for i, out in enumerate(outputs)}

def get_best_output(prompt, outputs, raw_image, model, processor):
    """ Identify the output with the highest probability """
    probs = prob_outs_given_prompt(prompt, outputs, raw_image, model, processor)
    return max(probs, key=probs.get)


def has_more_than_one_word(s):
    word_count = 0
    for word in s.split():
        word_count += 1
        if word_count > 1:
            return True
    return False








parser = argparse.ArgumentParser(description='Test different models on the server with specified GPU.')
parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use.')
parser.add_argument('--data_name', type=str, default='', help='Data path')
parser.add_argument('--local-rank', type=int, default=4, help='Local rank for distributed training')
parser.add_argument('--use_ddp', action='store_true', help='Flag to use Distributed Data Parallel')
parser.add_argument('--new_resolution', type=str, nargs='+', default='all', help='List of Resolution')
parser.add_argument('--dest_name', type=str, default='', help='Data path')


args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
model = AutoModelForCausalLM.from_pretrained(
    'THUDM/cogvlm-chat-hf',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()


dataset_paths = []

if args.data_name == 'icon' or args.data_name == 'all':
    dataset_path = '/homes/55/arshia/illusion-diffusion/illusion_generation/icon'
    dataset_paths.append(('icon',dataset_path))


if args.data_name == 'logo' or args.data_name == 'all':
    dataset_path = '/homes/55/arshia/illusion-diffusion/illusion_generation/logo'
    dataset_paths.append(('logo',dataset_path))
    

if args.data_name == 'sin' or args.data_name == 'all':
    dataset_path = '/homes/55/arshia/illusion-diffusion/illusion_generation/sin'
    dataset_paths.append(('sin',dataset_path))


    # dataset_path = '/homes/55/arshia/illusion-diffusion/Sin/ICON'
if args.data_name == 'old_sin' or args.data_name == 'all':
    dataset_path = '/homes/55/arshia/illusion-diffusion/cue-conflict'
    dataset_paths.append(('old_sin',dataset_path))




for each_datapath in dataset_paths:
    data_name = each_datapath[0]
    dataset_path = each_datapath[1]

    print(each_datapath,'starttttttt')


    classes = ['adidas', 'amazon','apple', 'audi', 'bmw', 'mercedes benz', 'facebook', 'google', 'ibm', 'instagram', 'linkedin', 'mcdonalds', 'nasa', 'nike', 'olympics', 'pepsi', 'playstation', 'puma', 'reebok', 'spotify', 'starbucks', 'tesla', 'telegram', 'ubuntu','Ocean', 'Origami', 'Forest', 'Cloud', 'Sand_dune','Medieval_Village', 'City', 'Underwater_ruins', 'Museum', 'Bazaar_market', 'Time_square']


    total_results = []







    old_sin_target_classes = ['airplane', 'bear', 'bicycle', 'bird', 'boat', 'bottle', 'car', 'cat', 'chair', 'clock', 'dog', 'elephant', 'keyboard', 'knife', 'oven', 'truck']
    sin_classes = ['Airplane', 'Bicycle', 'Bird', 'Bottle', 'Car', 'Cat', 'Dog', 'Dolphin', 'Fork', 'Guitar', 'Mug', 'Panda', 'Paper_clip', 'Sailboat', 'Scooter', 'Teapot']




    logo_classes = ['Adidas', 'Amazon','Apple', 'Audi', 'BMW', 'Mercedes Benz', 'Facebook', 'Google', 'Instagram', 'Mcdonalds', 'Nasa', 'Nike', 'Olympics', 'Playstation', 'Puma', 'Reebok', 'Spotify', 'Starbucks', 'Tesla', 'Telegram', 'Ubuntu']
    Icon_dataset = ['Animal', 'Face_Emoji', 'Music', 'Sport', 'Stationery', 'Vehicle']


    simple_prompts = ['Ocean', 'Origami', 'Forest', 'Cloud', 'Sand_dune']
    complex_prompts = ['Medieval_Village', 'City', 'Underwater_ruins', 'Museum', 'Bazaar_market', 'Time_square']

    simple_string = ', '.join(simple_prompts)
    complex_string = ', '.join(complex_prompts)
    old_sin_target_string = ','.join(old_sin_target_classes)
    sin_string = ', '.join(sin_classes)
    logo_string = ', '.join(logo_classes)
    icon_string = ', '.join(Icon_dataset)


    sin_prompts = [
        f'This image contains a icon integrated into a background, where elements of the background contribute to forming the icon. Identify the shape that is represented in the image by choosing exclusively among the following options:{sin_string},{simple_string}, {complex_string} Provide your response by stating only the single, most accurate class name that represents the icon. You have to respond with a single word.',
        f'This image contains an icon integrated into a background, where elements of the background contribute to forming the icon. Identify the background that is represented in the image by choosing exclusively among the following options:{sin_string},{simple_string}, {complex_string}. Provide your response by stating only the single, most accurate class name that represents the background. You have to respond with a single word.'


    ]


    icon_prompts = [
        f'This image contains a icon integrated into a background, where elements of the background contribute to forming the icon. Identify the icon that is represented in the image by choosing exclusively among the following options:{icon_string},{simple_string}, {complex_string} Provide your response by stating only the single, most accurate class name that represents the icon. You have to respond with a single word.',
        f'This image contains an icon integrated into a background, where elements of the background contribute to forming the icon. Identify the background that is represented in the image by choosing exclusively among the following options:{icon_string},{simple_string}, {complex_string}. Provide your response by stating only the single, most accurate class name that represents the background. You have to respond with a single word.'


    ]

    logo_prompts = [
        
        f'This image contains a icon integrated into a background, where elements of the background contribute to forming the logo. Identify the logo that is represented in the image by choosing exclusively among the following options:{logo_string},{simple_string}, {complex_string} Provide your response by stating only the single, most accurate class name that represents the logo. You have to respond with a single word.',
        f'This image contains an icon integrated into a background, where elements of the background contribute to forming the logo. Identify the background that is represented in the image by choosing exclusively among the following options:{logo_string},{simple_string}, {complex_string}. Provide your response by stating only the single, most accurate class name that represents the background. You have to respond with a single word.'
    
    ]


    old_sin_prompts = [
    
        
        f'This image contains a icon integrated into a background, where elements of the background contribute to forming the icon. Identify the icon that is represented in the image by choosing exclusively among the following options:{old_sin_target_string},{simple_string}, {complex_string} Provide your response by stating only the single, most accurate class name that represents the icon. You have to respond with a single word.',
        f'This image contains an icon integrated into a background, where elements of the background contribute to forming the icon. Identify the background that is represented in the image by choosing exclusively among the following options:{old_sin_target_string},{simple_string}, {complex_string}. Provide your response by stating only the single, most accurate class name that represents the background. You have to respond with a single word.'
    
    ]

    if data_name == 'old_sin':
        prompts = old_sin_prompts
    elif data_name == 'icon':
        prompts = icon_prompts
    elif data_name == 'logo':
        prompts = logo_prompts
    elif data_name == 'sin':
        prompts = sin_prompts



    not_single_words = []

    for each_prompt in prompts:

        normal_results = {}
        prob_results = {}

        not_single_word = 0

        print("START")
        

        for root, dirs, files in os.walk(dataset_path):
            path_parts = root.split(os.sep)
            if len(path_parts) > 1:
                print(path_parts)  # This checks if the path is at least one level deep within dataset_path
                if data_name == 'old_sin':
                    class_name = class_name = os.path.basename(root)
                else:
                    class_name = path_parts[-2]  # Get the first directory name after dataset_path
                for file in files:
                    if file.endswith('.png'):
                        if class_name not in normal_results:
                            normal_results[class_name] = {'shape': 0, 'texture': 0, 'rest': 0}
                            prob_results[class_name] = {'shape': 0, 'texture': 0, 'rest': 0}

                        image_path = os.path.join(root, file)
                        raw_image = Image.open(image_path).convert('RGB')
                        raw_image = raw_image.resize((int(args.new_resolution[0]),int(args.new_resolution[1])), Image.LANCZOS)
                        prompt = each_prompt
                        prediction = model_response(tokenizer, model,raw_image, prompt, device, 5, False)

                        # prob_class = get_best_output(prompt, classes, raw_image, model, processor,device)
                        if data_name == 'icon':
                            neg_class = file.split('-')[2]
                        elif data_name == 'old_sin':
                            neg_class = file.split('-')[1].split('.')[0][:-1]               
                        else:
                            neg_class = file.split('-')[1]

                        
                        print(f'actual: {class_name}, neg_class: {neg_class}, prediction: {prediction} ')
                        print(f'file: {file}, image_size: {raw_image.size}')
                        print('-----------------------------------------------')

                        if has_more_than_one_word(prediction):
                            not_single_word+=1

                        neg_class_white_space = neg_class.replace('_', ' ')
                        if class_name.lower() in prediction:
                            normal_results[class_name]['shape'] += 1
                        elif neg_class.lower() in prediction or neg_class_white_space in prediction:
                            normal_results[class_name]['texture'] += 1
                        else:
                            normal_results[class_name]['rest'] += 1

        not_single_words.append(not_single_word)
        total_results.append((each_prompt,normal_results))                    
            

    with open(f'/homes/55/arshia/illusion-diffusion/code/Model_evaluator/pickles/{data_name}/results_cogvlm_{args.dest_name}.pkl', 'wb') as pkl_file:
        pickle.dump(total_results, pkl_file)

    print(not_single_words)





