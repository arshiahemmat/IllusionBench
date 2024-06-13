
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
import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import AutoModel

# Initialize the process grou
# Specify the GPU to use





def model_response(processor, model,raw_image, prompt, device, num_beams=5, llava1p6_mode=False, llava1p5_mode=False):
    if llava1p6_mode:
        inputs = processor(prompt, raw_image, return_tensors="pt").to(device)
        if hasattr(model, 'module'):
            output = model.module.generate(
             **inputs, max_new_tokens=100
            )
        else: 
            output = model.generate(**inputs, max_new_tokens=100)
        input_string = processor.decode(output[0], skip_special_tokens=True)
        match = re.search(r"\[/INST\](.*)", input_string, re.IGNORECASE)


        if match:
            result = match.group(1).strip()  # Extract and strip the matched group
        else:
            result = "No content after [/INST]"

        print('---')
        return result.lower()

    elif llava1p5_mode:
        inputs = processor(prompt, raw_image, return_tensors="pt").to(device)
        if hasattr(model, 'module'):
            output = model.module.generate(
             **inputs, max_new_tokens=100, do_sample=False
            )
        else: 
            output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        input_string = processor.decode(output[0][2:], skip_special_tokens=True)
        match = re.search(r"assistant:\s*(\w+)", input_string.lower())
        if match:
            print('---')
            return match.group(1).lower()
        return None


    else:
        inputs = processor(raw_image, prompt, return_tensors="pt").to(device)
    
    if hasattr(model, 'module'):
        generated_ids = model.module.generate(
            **inputs, 
        max_length=200,
        num_beams=num_beams,       # Use 5 beams for the beam search
        early_stopping=True # Stop as soon as all beams find the EOS token
        )
    else: 
        generated_ids = model.generate(
                **inputs, 
                max_length=200,
                num_beams=num_beams,       # Use 5 beams for the beam search
                early_stopping=True # Stop as soon as all beams find the EOS token
            )    

    
    



    prediction = processor.decode(generated_ids[0], skip_special_tokens=True).strip()

    match = re.search(r"'([A-Z]+)'+\)", prediction)
    match1 = re.search(r"'([A-Z]+)'", prediction)
    match2 = re.search(r"([A-Z]+)\)", prediction)
    match3 = re.search(r"\[/INST\](.*)", prediction, re.IGNORECASE)

    if match:
        prediction = match.group(1)
    elif match1:
        prediction = match1.group(1)
    elif match2:
        prediction = match2.group(1)
    elif match3:
        prediction = match3.group(1).strip()
    
    # print(prediction)
    # print('---------')

    return prediction.lower()

def softmax(x):
    return torch.exp(x - torch.max(x)) / torch.sum(torch.exp(x - torch.max(x)))

def get_probabilities(prompt, outputs, raw_image, model, processor, device):
    log_probs = []
    detailed_debug_info = {}
    for output in outputs:
        input_dict = processor(raw_image, prompt, return_tensors='pt').to(device)
        output_dict = processor(raw_image, output, return_tensors='pt').to(device)

        pixel_values = input_dict['pixel_values']
        qformer_input_ids = input_dict['qformer_input_ids']

        with torch.no_grad():
            logits = model(input_ids=input_dict['input_ids'], attention_mask=input_dict['attention_mask'], 
                           pixel_values=pixel_values, qformer_input_ids=qformer_input_ids).logits
        
        output_token_prob = torch.log_softmax(logits[:, -1, :], dim=-1)[0, output_dict['input_ids'][0, -1]]
        log_probs.append(output_token_prob.item())

        # Debugging output
        detailed_debug_info[output] = {
            'log_prob': output_token_prob.item(),
            'predicted_tokens': processor.decode(logits.argmax(dim=-1)[0])
        }

    probabilities = softmax(torch.tensor(log_probs))
    probability_mapping = dict(zip(outputs, probabilities))

    # Print or log detailed debug info for analysis
    # .for key, value in detailed_debug_info.items():
        # print(f"Output: {key}, Log Prob: {value['log_prob']}, Predicted Tokens: {value['predicted_tokens']}")

    return probability_mapping

# Use this modified function in your prediction routine to get more insight.

def get_best_output(prompt, outputs, raw_image, model, processor, device):
    probabilities = get_probabilities(prompt, outputs, raw_image, model, processor, device)
    return max(probabilities, key=probabilities.get)

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
parser.add_argument('--model_name', type=str, default= 'v2', help='Please specify the BLIP version, choose between v2 and instruct')
parser.add_argument('--new_resolution', type=str, nargs='+', default='all', help='List of Resolution')
parser.add_argument('--dest_name', type=str, default='', help='Data path')

args = parser.parse_args()

device = torch.device(f"cuda:{args.local_rank}" if args.use_ddp and torch.cuda.is_available() else f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)  # Explicitly set the device
print(f"Using CUDA device: {device}")

if 'llava1.5' in args.model_name:
    from Llava_model import llava_find_model
    processor, model, template = llava_find_model(args.model_name)
    model.to(device)    
elif 'MoE' in args.model_name:
    from moe import find_ans
    template = '[ORGINAL]'
else:
    from Model import find_model
    processor, model, template = find_model(args.model_name)
    model.to(device)




if args.use_ddp:
    # Initialize the process group for Distributed Data Parallel
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend='nccl', rank=args.local_rank, init_method='env://')

# Set the device based on local_rank if using DDP, or a specified GPU otherwise


# Transfer the model to the configured device

# Wrap the model with DistributedDataParallel if using DDP
if torch.cuda.is_available() and torch.distributed.is_initialized():
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    print('Multi-GPU has been started...')


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




# image_processor = processor.image_processor
# image_processor.size['shortest_edge'] = int(args.new_resolution[0])
# image_processor.crop_size = {'height': int(args.new_resolution[0]), 'width': int(args.new_resolution[1])}


total_response = {}

for each_datapath in dataset_paths:
    data_name = each_datapath[0]
    dataset_path = each_datapath[1]

    print(each_datapath,'starttttttt')


    classes = ['adidas', 'amazon','apple', 'audi', 'bmw', 'mercedes benz', 'facebook', 'google', 'ibm', 'instagram', 'linkedin', 'mcdonalds', 'nasa', 'nike', 'olympics', 'pepsi', 'playstation', 'puma', 'reebok', 'spotify', 'starbucks', 'tesla', 'telegram', 'ubuntu','Ocean', 'Origami', 'Forest', 'Cloud', 'Sand_dune','Medieval_Village', 'City', 'Underwater_ruins', 'Museum', 'Bazaar_market', 'Time_square']


    total_results = []

    total_response[each_datapath] = {}



    old_sin_target_classes = ['airplane', 'bear', 'bicycle', 'bird', 'boat', 'bottle', 'car', 'cat', 'chair', 'clock', 'dog', 'elephant', 'keyboard', 'knife', 'oven', 'truck']
    sin_classes = ['Airplane', 'Bicycle', 'Bird', 'Bottle', 'Car', 'Cat', 'Dog', 'Dolphin', 'Fork', 'Guitar', 'Mug', 'Panda', 'Paper_clip', 'Sailboat', 'Scooter', 'Teapot']




    logo_classes = ['Adidas', 'Amazon','Apple', 'Audi', 'BMW', 'Mercedes Benz', 'Facebook', 'Google', 'Instagram', 'Mcdonalds', 'Nasa', 'Nike', 'Olympics', 'Playstation', 'Puma', 'Reebok', 'Spotify', 'Starbucks', 'Tesla', 'Telegram', 'Ubuntu']
    Icon_dataset = ['Animal', 'Face_Emoji', 'Music', 'Sport', 'Stationery', 'Vehicle']


    simple_prompts = ['Ocean', 'Origami', 'Forest', 'Cloud', 'Sand_dune']
    complex_prompts = ['Medieval_Village', 'City', 'Underwater_ruins', 'Museum', 'Bazaar_market', 'Time_square']
    background_prompts = ['Ocean', 'Origami', 'Forest', 'Cloud', 'Sand_dune','Medieval_Village', 'City', 'Underwater_ruins', 'Museum', 'Bazaar_market', 'Time_square']

    simple_string = ', '.join(simple_prompts)
    complex_string = ', '.join(complex_prompts)
    background_string = ', '.join(background_prompts)
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

    for idx, each_prompt in enumerate(prompts):

        normal_results = {}
        prob_results = {}


        total_response[each_datapath][idx+1] = {}



        not_single_word = 0

        print("START")
        new_prompt = template.replace('[ORGINAL]',each_prompt)
        

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
                        prompt = new_prompt
                        if 'MoE' in args.model_name:
                            if 'qwen' in args.model_name:
                                prediction = find_ans('qwen', prompt, image_path,device)
                            
                            if 'phi2' in args.model_name:
                                prediction = find_ans('phi2', prompt, image_path,device)
                            
                            if 'stable' in args.model_name:
                                prediction = find_ans('stable', prompt, image_path,device)
                            
                        else:
                            prediction = model_response(processor, model,raw_image, prompt, device, 5, 'llava1.6' in args.model_name, 'llava1.5' in args.model_name)
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

                        total_response[each_datapath][idx+1][file] = [class_name,neg_class,prediction]


                        print(total_response)


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
            

    with open(f'/homes/55/arshia/illusion-diffusion/code/Model_evaluator/pickles/{data_name}/results_{args.model_name}_{args.dest_name}.pkl', 'wb') as pkl_file:
        pickle.dump(total_results, pkl_file)

    print(not_single_words)


with open(f'/homes/55/arshia/illusion-diffusion/code/Model_evaluator/responses/result_{args.model_name}_{args.dest_name}.pkl', 'wb') as pkl_file:
    pickle.dump(total_response, pkl_file)





