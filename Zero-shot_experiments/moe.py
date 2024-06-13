from PIL import Image
import torch

class MoE_LLaVAModel():

    def __init__(self, model_name,device):
        from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from moellava.model.builder import load_pretrained_model
        from moellava.utils import disable_torch_init
        from moellava.mm_utils import get_model_name_from_path

        disable_torch_init()
        self.tokenizer, self.model, processor, _ = load_pretrained_model(
            model_name,
            None,
            get_model_name_from_path(model_name),
            device_map=device,
        )
        self.model.eval()
        self.image_processor = processor["image"]

        conv_mode = None
        if "stablelm" in model_name.lower():
            conv_mode = "stablelm"
        elif "phi" in model_name.lower():
            conv_mode = "phi"
        elif "qwen" in model_name.lower():
            conv_mode = "qwen"
        else:
            raise ValueError(f"Unknown conversation {model_name}")

        self.conv_mode = conv_mode
        self.temperature = 0.2

    @staticmethod
    def get_MoE_LLaVA_StableLM(device):
        return MoE_LLaVAModel("LanguageBind/MoE-LLaVA-StableLM-1.6B-4e",device)

    @staticmethod
    def get_MoE_LLaVA_Qwen(device):
        return MoE_LLaVAModel("LanguageBind/MoE-LLaVA-Qwen-1.8B-4e",device)

    @staticmethod
    def get_MoE_LLaVA_Phi2(device):
        return MoE_LLaVAModel("LanguageBind/MoE-LLaVA-Phi2-2.7B-4e",device)

    @staticmethod
    def get_MoE_LLaVA_Phi2_384(device):
        return MoE_LLaVAModel("LanguageBind/MoE-LLaVA-Phi2-2.7B-4e-384",device)

    def forward(self, prompt: str, image_path: str) -> dict:
        from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from moellava.conversation import conv_templates, SeparatorStyle
        from moellava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

        image = Image.open(image_path).convert("RGB")

        conv = conv_templates[self.conv_mode].copy()
        image_tensor = self.image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ].to(self.model.device, dtype=torch.float16)

        inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.model.device)
        )
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1] :], skip_special_tokens=True
        ).strip()
        return outputs.lower()


def find_ans(model, prompt, image_path,device):
    if model == 'qwen':
        vl_model = MoE_LLaVAModel.get_MoE_LLaVA_Qwen(device)
        return vl_model.forward(prompt,image_path)
    elif model == 'phi2':
        vl_model = MoE_LLaVAModel.get_MoE_LLaVA_Phi2(device)
        return vl_model.forward(prompt,image_path)
    else:
        vl_model = MoE_LLaVAModel.get_MoE_LLaVA_StableLM(device)
        return vl_model.forward(prompt,image_path)