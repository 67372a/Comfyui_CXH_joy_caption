
from huggingface_hub import InferenceClient
from torch import nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM
from pathlib import Path
import torch
import torch.amp.autocast_mode
from PIL import Image
import os
import folder_paths
import torchvision.transforms.functional as TVF
from peft import PeftConfig
import datetime
import hashlib

from .lib.ximg import *
from .lib.xmodel import *

from model_management import get_torch_device

DEVICE = get_torch_device()
# def get_torch_device():  
#     """  
#     返回PyTorch模型应该运行的设备（CPU或GPU）  
#     如果系统支持CUDA并且至少有一个GPU可用，则返回GPU设备；否则返回CPU设备。  
#     """  
#     if torch.cuda.is_available():  
#         # 选择第一个可用的GPU  
#         device = torch.device("cuda:0")  
#         print(f"There are {torch.cuda.device_count()} GPU(s) available.")  
#         print(f"We will use the GPU: {device}")  
#     else:  
#         # 如果没有GPU可用，则使用CPU  
#         device = torch.device("cpu")  
#         print("No GPU available, using the CPU instead.")  
#     return device

class JoyPipeline:
    def __init__(self):
        self.clip_model = None
        self.clip_processor =None
        self.tokenizer = None
        self.text_model = None
        self.image_adapter = None
        self.parent = None
    
    def clearCache(self):
        self.clip_model = None
        self.clip_processor =None
        self.tokenizer = None
        self.text_model = None
        self.image_adapter = None 


class ImageAdapter(nn.Module):
	def __init__(self, input_features: int, output_features: int):
		super().__init__()
		self.linear1 = nn.Linear(input_features, output_features)
		self.activation = nn.GELU()
		self.linear2 = nn.Linear(output_features, output_features)
	
	def forward(self, vision_outputs: torch.Tensor):
		x = self.linear1(vision_outputs)
		x = self.activation(x)
		x = self.linear2(x)
		return x

class ImageAdapterAlphaOne(nn.Module):
	def __init__(self, input_features: int, output_features: int, ln1: bool, pos_emb: bool, num_image_tokens: int, deep_extract: bool):
		super().__init__()
		self.deep_extract = deep_extract

		if self.deep_extract:
			input_features = input_features * 5

		self.linear1 = nn.Linear(input_features, output_features)
		self.activation = nn.GELU()
		self.linear2 = nn.Linear(output_features, output_features)
		self.ln1 = nn.Identity() if not ln1 else nn.LayerNorm(input_features)
		self.pos_emb = None if not pos_emb else nn.Parameter(torch.zeros(num_image_tokens, input_features))

		# Mode token
		#self.mode_token = nn.Embedding(n_modes, output_features)
		#self.mode_token.weight.data.normal_(mean=0.0, std=0.02)   # Matches HF's implementation of llama3

		# Other tokens (<|image_start|>, <|image_end|>, <|eot_id|>)
		self.other_tokens = nn.Embedding(3, output_features)
		self.other_tokens.weight.data.normal_(mean=0.0, std=0.02)   # Matches HF's implementation of llama3

	def forward(self, vision_outputs: torch.Tensor):
		if self.deep_extract:
			x = torch.concat((
				vision_outputs[-2],
				vision_outputs[3],
				vision_outputs[7],
				vision_outputs[13],
				vision_outputs[20],
			), dim=-1)
			assert len(x.shape) == 3, f"Expected 3, got {len(x.shape)}"  # batch, tokens, features
			assert x.shape[-1] == vision_outputs[-2].shape[-1] * 5, f"Expected {vision_outputs[-2].shape[-1] * 5}, got {x.shape[-1]}"
		else:
			x = vision_outputs[-2]

		x = self.ln1(x)

		if self.pos_emb is not None:
			assert x.shape[-2:] == self.pos_emb.shape, f"Expected {self.pos_emb.shape}, got {x.shape[-2:]}"
			x = x + self.pos_emb

		x = self.linear1(x)
		x = self.activation(x)
		x = self.linear2(x)

		# Mode token
		#mode_token = self.mode_token(mode)
		#assert mode_token.shape == (x.shape[0], mode_token.shape[1], x.shape[2]), f"Expected {(x.shape[0], 1, x.shape[2])}, got {mode_token.shape}"
		#x = torch.cat((x, mode_token), dim=1)

		# <|image_start|>, IMAGE, <|image_end|>
		other_tokens = self.other_tokens(torch.tensor([0, 1], device=self.other_tokens.weight.device).expand(x.shape[0], -1))
		assert other_tokens.shape == (x.shape[0], 2, x.shape[2]), f"Expected {(x.shape[0], 2, x.shape[2])}, got {other_tokens.shape}"
		x = torch.cat((other_tokens[:, 0:1], x, other_tokens[:, 1:2]), dim=1)

		return x

	def get_eot_embedding(self):
		return self.other_tokens(torch.tensor([2], device=self.other_tokens.weight.device)).squeeze(0)

class Joy_caption_load:

    def __init__(self):
        self.model = None
        self.pipeline = JoyPipeline()
        self.pipeline.parent = self
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["unsloth/Meta-Llama-3.1-8B-bnb-4bit", "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit","meta/Meta-Llama-3.1-8B","meta/Meta-Llama-3.1-8B-Instruct", "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"],), 
               
            }
        }

    CATEGORY = "CXH/LLM"
    RETURN_TYPES = ("JoyPipeline",)
    FUNCTION = "gen"

    def loadCheckPoint(self):
        # 清除一波
        if self.pipeline != None:
            self.pipeline.clearCache() 
       
         # clip
        model_id = "google/siglip-so400m-patch14-384"
        CLIP_PATH = download_hg_model(model_id,"clip")

        clip_processor = AutoProcessor.from_pretrained(CLIP_PATH) 
        clip_model = AutoModel.from_pretrained(
                CLIP_PATH,
                trust_remote_code=True
            )
            
        clip_model = clip_model.vision_model
        clip_model.eval()
        clip_model.requires_grad_(False)
        clip_model.to(DEVICE)

       
        # LLM
        MODEL_PATH = download_hg_model(self.model,"LLM")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,use_fast=False)
        assert isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer, PreTrainedTokenizerFast), f"Tokenizer is of type {type(tokenizer)}"

        text_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto",trust_remote_code=True)
        text_model.eval()

        # Image Adapter
        adapter_path =  os.path.join(folder_paths.models_dir,"Joy_caption","image_adapter.pt")

        image_adapter = ImageAdapter(clip_model.config.hidden_size, text_model.config.hidden_size) # ImageAdapter(clip_model.config.hidden_size, 4096) 
        image_adapter.load_state_dict(torch.load(adapter_path, map_location="cpu"))
        adjusted_adapter =  image_adapter #AdjustedImageAdapter(image_adapter, text_model.config.hidden_size)
        adjusted_adapter.eval()
        adjusted_adapter.to(DEVICE)

        self.pipeline.clip_model = clip_model
        self.pipeline.clip_processor = clip_processor
        self.pipeline.tokenizer = tokenizer
        self.pipeline.text_model = text_model
        self.pipeline.image_adapter = adjusted_adapter
    
    def clearCache(self):
         if self.pipeline != None:
              self.pipeline.clearCache()

    def gen(self,model):
        if self.model == None or self.model != model or self.pipeline == None:
            self.model = model
            self.loadCheckPoint()
        return (self.pipeline,)

class Joy_caption_load_alpha_one:

    def __init__(self):
        self.model = None
        self.pipeline = JoyPipeline()
        self.pipeline.parent = self
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["unsloth/Meta-Llama-3.1-8B-bnb-4bit", "meta/Meta-Llama-3.1-8B"],), 
               
            }
        }

    CATEGORY = "CXH/LLM"
    RETURN_TYPES = ("JoyPipeline",)
    FUNCTION = "gen"

    def loadCheckPoint(self):
        # 清除一波
        if self.pipeline != None:
            self.pipeline.clearCache() 
       
         # clip
        model_id = "google/siglip-so400m-patch14-384"
        CLIP_PATH = download_hg_model(model_id,"clip")

        clip_processor = AutoProcessor.from_pretrained(CLIP_PATH) 
        clip_model = AutoModel.from_pretrained(
                CLIP_PATH,
                trust_remote_code=True
            )
            
        joycaption_path = os.path.join(folder_paths.models_dir,"Joy_caption_alpha_one")
        clip_model = clip_model.vision_model
        if os.path.exists(os.path.join(joycaption_path,"clip_model.pt")):
            print("Loading VLM's custom vision model")
            checkpoint = torch.load(os.path.join(joycaption_path,"clip_model.pt"), map_location='cpu')
            checkpoint = {k.replace("_orig_mod.module.", ""): v for k, v in checkpoint.items()}
            clip_model.load_state_dict(checkpoint)
            del checkpoint
            
        clip_model.eval()
        clip_model.requires_grad_(False)
        clip_model.to(DEVICE)
        
        MODEL_PATH = download_hg_model(self.model,"LLM")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,use_fast=False)
        assert isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer, PreTrainedTokenizerFast), f"Tokenizer is of type {type(tokenizer)}"

        # LLM
        text_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", trust_remote_code=True)
        
        if os.path.exists(os.path.join(joycaption_path,"text_model")):
            print("Loading VLM's custom text model lora")
            loraConfig = PeftConfig.from_pretrained(os.path.join(joycaption_path,"text_model"))
            loraConfig.base_model_name_or_path=self.model
            text_model.add_adapter(loraConfig, os.path.join(joycaption_path,"text_model"))
            
        text_model.eval()
        

        # Image Adapter
        adapter_path =  os.path.join(folder_paths.models_dir,"Joy_caption_alpha_one","image_adapter.pt")

        image_adapter = ImageAdapterAlphaOne(clip_model.config.hidden_size, text_model.config.hidden_size, False, False, 38, False) # ImageAdapter(clip_model.config.hidden_size, 4096) 
        image_adapter.load_state_dict(torch.load(adapter_path, map_location="cpu"))
        adjusted_adapter =  image_adapter #AdjustedImageAdapter(image_adapter, text_model.config.hidden_size)
        adjusted_adapter.eval()
        adjusted_adapter.to(DEVICE)

        self.pipeline.clip_model = clip_model
        self.pipeline.clip_processor = clip_processor
        self.pipeline.tokenizer = tokenizer
        self.pipeline.text_model = text_model
        self.pipeline.image_adapter = adjusted_adapter
    
    def clearCache(self):
         if self.pipeline != None:
              self.pipeline.clearCache()

    def gen(self,model):
        if self.model == None or self.model != model or self.pipeline == None:
            self.model = model
            self.loadCheckPoint()
        return (self.pipeline,)

class Joy_caption:
    original_IS_CHANGED = None

    def __init__(self):
        self.reroll_result = "enable"
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "joy_pipeline": ("JoyPipeline",),
                "image": ("IMAGE",),
                "prompt":   ("STRING", {"multiline": True, "default": "A descriptive caption for this image:\n"},),
                "max_new_tokens":("INT", {"default": 1024, "min": 10, "max": 4096, "step": 1}),
                "temperature": ("FLOAT", {"default": 0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 200, "step": 1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0, "step": 0.01}),
                "beams": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
                "length_penalty": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "reroll_result": (["enable", "disable"], {"default": "enable"}),
                "cache_models": ("BOOLEAN", {"default": True}),
            }
        }

    CATEGORY = "CXH/LLM"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "gen"
    def gen(self,joy_pipeline,image,prompt,max_new_tokens,temperature,top_k,top_p,beams,length_penalty,reroll_result,cache_models): 
    
        self.reroll_result = reroll_result
        if Joy_caption.original_IS_CHANGED is None:
            Joy_caption.original_IS_CHANGED = Joy_caption.IS_CHANGED
        if self.reroll_result == "enable":
            setattr(Joy_caption, "IS_CHANGED", Joy_caption.original_IS_CHANGED)
        else:
            if hasattr(Joy_caption, "IS_CHANGED"):
                delattr(Joy_caption, "IS_CHANGED")
    
        if joy_pipeline.clip_processor == None:
            joy_pipeline.parent.loadCheckPoint()    

        clip_processor = joy_pipeline.clip_processor
        tokenizer = joy_pipeline.tokenizer
        clip_model = joy_pipeline.clip_model
        image_adapter = joy_pipeline.image_adapter
        text_model = joy_pipeline.text_model

     

        input_image = tensor2pil(image)

        # Preprocess image
        pImge = clip_processor(images=input_image, return_tensors='pt').pixel_values
        pImge = pImge.to(DEVICE)

        # Tokenize the prompt
        prompt = tokenizer.encode(prompt, return_tensors='pt', padding=False, truncation=False, add_special_tokens=False)
        # Embed image
        with torch.amp.autocast_mode.autocast(str(DEVICE), enabled=True):
            vision_outputs = clip_model(pixel_values=pImge, output_hidden_states=True)
            image_features = vision_outputs.hidden_states[-2]
            embedded_images = image_adapter(image_features)
            embedded_images = embedded_images.to(DEVICE)

        # Embed prompt
        prompt_embeds = text_model.model.embed_tokens(prompt.to(DEVICE))
        assert prompt_embeds.shape == (1, prompt.shape[1], text_model.config.hidden_size), f"Prompt shape is {prompt_embeds.shape}, expected {(1, prompt.shape[1], text_model.config.hidden_size)}"
        embedded_bos = text_model.model.embed_tokens(torch.tensor([[tokenizer.bos_token_id]], device=text_model.device, dtype=torch.int64))   

        # Construct prompts
        inputs_embeds = torch.cat([
            embedded_bos.expand(embedded_images.shape[0], -1, -1),
            embedded_images.to(dtype=embedded_bos.dtype),
            prompt_embeds.expand(embedded_images.shape[0], -1, -1),
        ], dim=1)

        input_ids = torch.cat([
            torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long),
            torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
            prompt,
        ], dim=1).to(DEVICE)
        attention_mask = torch.ones_like(input_ids)
        
        do_sample=True
        if temperature == 0:
            do_sample=False
        
        generate_ids = text_model.generate(input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature, suppress_tokens=None, num_beams=beams, length_penalty=length_penalty)

        # Trim off the prompt
        generate_ids = generate_ids[:, input_ids.shape[1]:]
        if generate_ids[0][-1] == tokenizer.eos_token_id:
            generate_ids = generate_ids[:, :-1]

        caption = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
        r = caption.strip()

        if cache_models == False:
           joy_pipeline.parent.clearCache()

        return (r,)
        
    @classmethod
    def IS_CHANGED(s):
        hash_value = hashlib.md5(str(datetime.datetime.now()).encode()).hexdigest()
        return hash_value

class Joy_caption_alpha_one:
    original_IS_CHANGED = None

    def __init__(self):
        self.reroll_result = "enable"
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "joy_pipeline": ("JoyPipeline",),
                "image": ("IMAGE",),
                "prompt":   ("STRING", {"multiline": True, "default": "A descriptive caption for this image:\n"},),
                "max_new_tokens":("INT", {"default": 1024, "min": 10, "max": 4096, "step": 1}),
                "temperature": ("FLOAT", {"default": 0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 200, "step": 1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0, "step": 0.01}),
                "beams": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
                "length_penalty": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "reroll_result": (["enable", "disable"], {"default": "enable"}),
                "cache_models": ("BOOLEAN", {"default": True}),
            }
        }

    CATEGORY = "CXH/LLM"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "gen"
    def gen(self,joy_pipeline,image,prompt,max_new_tokens,temperature,top_k,top_p,beams,length_penalty,reroll_result,cache_models): 
    
        self.reroll_result = reroll_result
        if Joy_caption.original_IS_CHANGED is None:
            Joy_caption.original_IS_CHANGED = Joy_caption.IS_CHANGED
        if self.reroll_result == "enable":
            setattr(Joy_caption, "IS_CHANGED", Joy_caption.original_IS_CHANGED)
        else:
            if hasattr(Joy_caption, "IS_CHANGED"):
                delattr(Joy_caption, "IS_CHANGED")
    
        if joy_pipeline.clip_processor == None:
            joy_pipeline.parent.loadCheckPoint()    

        clip_processor = joy_pipeline.clip_processor
        tokenizer = joy_pipeline.tokenizer
        clip_model = joy_pipeline.clip_model
        image_adapter = joy_pipeline.image_adapter
        text_model = joy_pipeline.text_model

        input_image = tensor2pil(image)

        # Preprocess image
        image = input_image.resize((384, 384), Image.LANCZOS)
        pixel_values = TVF.pil_to_tensor(image).unsqueeze(0) / 255.0
        pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
        pixel_values = clip_processor(images=image, return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(DEVICE)

        # Tokenize the prompt
        prompt = tokenizer.encode(prompt, return_tensors='pt', padding=False, truncation=False, add_special_tokens=False)
        
        # Embed image
        with torch.amp.autocast_mode.autocast(str(DEVICE), enabled=True):
            vision_outputs = clip_model(pixel_values=pixel_values, output_hidden_states=True)
            image_features = vision_outputs.hidden_states
            embedded_images = image_adapter(image_features)
            embedded_images = embedded_images.to(DEVICE)

        # Embed prompt
        prompt_embeds = text_model.model.embed_tokens(prompt.to(DEVICE))
        assert prompt_embeds.shape == (1, prompt.shape[1], text_model.config.hidden_size), f"Prompt shape is {prompt_embeds.shape}, expected {(1, prompt.shape[1], text_model.config.hidden_size)}"
        embedded_bos = text_model.model.embed_tokens(torch.tensor([[tokenizer.bos_token_id]], device=text_model.device, dtype=torch.int64))
        eot_embed = image_adapter.get_eot_embedding().unsqueeze(0).to(dtype=text_model.dtype)

        # Construct prompts
        inputs_embeds = torch.cat([
            embedded_bos.expand(embedded_images.shape[0], -1, -1),
            embedded_images.to(dtype=embedded_bos.dtype),
            prompt_embeds.expand(embedded_images.shape[0], -1, -1),
            eot_embed.expand(embedded_images.shape[0], -1, -1),
        ], dim=1)

        input_ids = torch.cat([
            torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long),
            torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
            prompt,
            torch.tensor([[tokenizer.convert_tokens_to_ids("<|eot_id|>")]], dtype=torch.long),
        ], dim=1).to(DEVICE)
        attention_mask = torch.ones_like(input_ids)

        do_sample=True
        if temperature == 0:
            do_sample=False
        
        generate_ids = text_model.generate(input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature, suppress_tokens=None, num_beams=beams, length_penalty=length_penalty)

        # Trim off the prompt
        generate_ids = generate_ids[:, input_ids.shape[1]:]
        if generate_ids[0][-1] == tokenizer.eos_token_id:
            generate_ids = generate_ids[:, :-1]

        caption = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
        r = caption.strip()

        if cache_models == False:
           joy_pipeline.parent.clearCache()

        return (r,)
        
    @classmethod
    def IS_CHANGED(s):
        hash_value = hashlib.md5(str(datetime.datetime.now()).encode()).hexdigest()
        return hash_value
