# from gradio_client import Client
# import gradio_client
import shutil
import os
import argparse
import torch
from PIL import Image
import random
import time
from diffusers import (
    DiffusionPipeline,
    AutoencoderKL,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    StableDiffusionLatentUpscalePipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    DPMSolverMultistepScheduler,  # <-- Added import
    EulerDiscreteScheduler  # <-- Added import
)





parser = argparse.ArgumentParser(description='Test different models on the server with specified GPU.')
parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use.')
parser.add_argument('--data_list', type=str, nargs='+', default='all', help='List of Classes')
parser.add_argument('--ex_mode', type= str, default='logo', help = 'logo, icon, and sin')
parser.add_argument('--prompts_type', type= str, default='simple', help = 'Simple or Complex')
parser.add_argument('--ex_type', type= str, default='simple', help = 'Simple or Complex')



args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")


BASE_MODEL = "SG161222/Realistic_Vision_V5.1_noVAE"

# Initialize both pipelines
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
#init_pipe = DiffusionPipeline.from_pretrained("SG161222/Realistic_Vision_V5.1_noVAE", torch_dtype=torch.float16)
controlnet = ControlNetModel.from_pretrained("monster-labs/control_v1p_sd15_qrcode_monster", torch_dtype=torch.float16)#, torch_dtype=torch.float16)
main_pipe = StableDiffusionControlNetPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnet,
    vae=vae,
    safety_checker=None,
    torch_dtype=torch.float16,
).to(device)


image_pipe = StableDiffusionControlNetImg2ImgPipeline(**main_pipe.components)



# Sampler map
SAMPLER_MAP = {
    "DPM++ Karras SDE": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True, algorithm_type="sde-dpmsolver++"),
    "Euler": lambda config: EulerDiscreteScheduler.from_config(config),
}

def center_crop_resize(img, output_size=(512, 512)):
    width, height = img.size

    # Calculate dimensions to crop to the center
    new_dimension = min(width, height)
    left = (width - new_dimension)/2
    top = (height - new_dimension)/2
    right = (width + new_dimension)/2
    bottom = (height + new_dimension)/2

    # Crop and resize
    img = img.crop((left, top, right, bottom))
    img = img.resize(output_size)

    return img

def common_upscale(samples, width, height, upscale_method, crop=False):
        if crop == "center":
            old_width = samples.shape[3]
            old_height = samples.shape[2]
            old_aspect = old_width / old_height
            new_aspect = width / height
            x = 0
            y = 0
            if old_aspect > new_aspect:
                x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
            elif old_aspect < new_aspect:
                y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
            s = samples[:,:,y:old_height-y,x:old_width-x]
        else:
            s = samples

        return torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)

def upscale(samples, upscale_method, scale_by):
        #s = samples.copy()
        width = round(samples["images"].shape[3] * scale_by)
        height = round(samples["images"].shape[2] * scale_by)
        s = common_upscale(samples["images"], width, height, upscale_method, "disabled")
        return (s)

def check_inputs(prompt: str, control_image: Image.Image):
    if control_image is None:
        raise gr.Error("Please select or upload an Input Illusion")
    if prompt is None or prompt == "":
        raise gr.Error("Prompt is required")

def convert_to_pil(base64_image):
    pil_image = Image.open(base64_image)
    return pil_image

def convert_to_base64(pil_image):
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        image.save(temp_file.name)
    return temp_file.name

# Inference function
def inference(
    control_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float = 8.0,
    controlnet_conditioning_scale: float = 1,
    control_guidance_start: float = 1,    
    control_guidance_end: float = 1,
    upscaler_strength: float = 0.5,
    seed: int = -1,
    sampler = "DPM++ Karras SDE",
    device = device
):
    start_time = time.time()
    start_time_struct = time.localtime(start_time)
    start_time_formatted = time.strftime("%H:%M:%S", start_time_struct)
    print(f"Inference started at {start_time_formatted}")

    control_image_small = center_crop_resize(control_image)
    control_image_large = center_crop_resize(control_image, (1024, 1024))

    main_pipe.scheduler = SAMPLER_MAP[sampler](main_pipe.scheduler.config)
    my_seed = random.randint(0, 2**32 - 1) if seed == -1 else seed
    generator = torch.Generator(device=device).manual_seed(my_seed)
    
    out = main_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control_image_small,
        guidance_scale=float(guidance_scale),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        generator=generator,
        control_guidance_start=float(control_guidance_start),
        control_guidance_end=float(control_guidance_end),
        num_inference_steps=15,
        output_type="latent"
    )
    upscaled_latents = upscale(out, "nearest-exact", 2)
    out_image = image_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        control_image=control_image_large,        
        image=upscaled_latents,
        guidance_scale=float(guidance_scale),
        generator=generator,
        num_inference_steps=20,
        strength=upscaler_strength,
        control_guidance_start=float(control_guidance_start),
        control_guidance_end=float(control_guidance_end),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale)
    )
    end_time = time.time()
    end_time_struct = time.localtime(end_time)
    end_time_formatted = time.strftime("%H:%M:%S", end_time_struct)
    print(f"Inference ended at {end_time_formatted}, taking {end_time-start_time}s")

    # Save image + metadata
   

    return out_image["images"][0]




if args.ex_mode == 'logo':
    icon_path = '/homes/55/arshia/illusion-diffusion/psycho_exp/Logo_icon'

if args.ex_mode == 'icon':
    icon_path = '/homes/55/arshia/illusion-diffusion/psycho_exp/Icon_icon'

if args.ex_mode == 'sin':
    icon_path = '/homes/55/arshia/illusion-diffusion/psycho_exp/Sin_plus_icon'

experiment_type = args.ex_type

roots = f'/homes/55/arshia/illusion-diffusion/illusion_generation/{experiment_type}/{args.ex_mode}'


# prompts = ['Underwater_ruins', 'Origami', 'Forest']
simple_prompts = ['Ocean', 'Origami', 'Forest', 'Cloud', 'Sand_dune']
complex_prompts = ['Medieval_Village', 'City', 'Underwater_ruins', 'Museum', 'Bazaar_market', 'Time_square']

all_prompts = ['Ocean', 'Origami', 'Forest', 'Cloud', 'Sand_dune','Medieval_Village', 'City', 'Underwater_ruins', 'Museum', 'Bazaar_market', 'Time_square']

gpt_prompts = ['Origami','Sand_dune','Medieval_Village','Bazaar_market']

if args.prompts_type == 'simple':
    if not os.path.exists(f'{roots}/simple_prompts'):
        os.makedirs(f'{roots}/simple_prompts')
    roots = f'/homes/55/arshia/illusion-diffusion/illusion_generation/{experiment_type}/{args.ex_mode}/simple_prompts'
    prompts = simple_prompts

elif args.prompts_type == 'complex':
    if not os.path.exists(f'{roots}/complex_prompts'):
        os.makedirs(f'{roots}/complex_prompts')
    roots = f'/homes/55/arshia/illusion-diffusion/illusion_generation/{experiment_type}/{args.ex_mode}/complex_prompts'
    prompts = complex_prompts

elif args.prompts_type == 'all':
    prompts = all_prompts

elif args.prompts_type == 'gpt':
    prompts = gpt_prompts








levels = ['Easy','Hard']


if True :
    values = {
        'Easy': [1.25, 1.40],
        'Hard': [1.05,0.85]
    }
else:
    values = {
        'Easy': [1.25, 1.35, 1.40 ,1.50, 1.60],
        'Hard': [1.20, 1.15 ,1.10, 1.05, .90, .85 , .80 , .75]
    }

categories = ['Animal','Face_Emoji','Music','Sport','Stationary','Vehicle']

image_count = {category: 0 for category in categories}

for root, dirs, files in os.walk(icon_path):
    for file in files:
        if file.endswith('.png'):
            # class_name = os.path.basename(root)
            class_name = os.path.basename(file).split('.')[0].split('-')[1]
            if class_name in args.data_list:
                continue
            identifier = os.path.basename(file).split('.')[0].split('-')[-1]
            if args.prompts_type != 'psycho':
                if not os.path.exists(f'{roots}/{class_name}'):
                    os.makedirs(f'{roots}/{class_name}')

            for each_prompts in prompts:
                for each_levels in levels:
                    if args.prompts_type != 'psycho': 
                        if not os.path.exists(f'{roots}/{class_name}/{each_levels}'):
                            os.makedirs(f'{roots}/{class_name}/{each_levels}')

                    
                    raw_image = Image.open(f'{root}/{file}')
                    
                    for each_values in values[each_levels]:
                        # if args.ex_mode == 'icon':
                        #     image_count[class_name] += 1
                        result = inference(
                                raw_image,   # input_illusion: filepath
                                each_prompts,                  # prompt: str
                                "low quality",                         # negative_prompt: str (assuming empty if not provided)
                                7.5,                        # guidance_scale: float
                                each_values,                        # illusion_strength: float
                                0.0,                        # start_of_controlnet: float (assuming 0 if not provided)
                                1.0,                        # end_of_controlnet: float
                                1.0,                        # strength_of_the_upscaler: float
                                -1,
                                "Euler",
                                device                  # seed: float       # Correct API endpoint
                            )
                        if args.prompts_type != 'all':
                            save_path = f'{roots}/{class_name}/{each_levels}/{class_name.lower()}-{each_prompts}-{each_levels}-{each_values}-{identifier}.png'
                            
                            if args.ex_mode == 'icon':
                                icon_save_path = f'{roots}/{class_name}/{each_levels}/{class_name.lower()}-{each_prompts}-{each_levels}-{each_values}-{identifier}.png'
                                save_path = icon_save_path
                        else:
                            save_path = f'{roots}/{class_name.lower()}-{each_prompts}-{each_levels}-{each_values}-{identifier}.png'


                        result.save(save_path, format='PNG')
                        print('image path:', save_path)








