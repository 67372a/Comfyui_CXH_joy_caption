
from huggingface_hub import InferenceClient
from torch import nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM
from pathlib import Path
import torch
import torch.amp.autocast_mode
from PIL import Image
import os
import folder_paths

from .lib.ximg import *
from .lib.xmodel import *

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
        clip_model.to('cuda:0')

       
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
        adjusted_adapter.to('cuda:0')

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
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 10, "min": 0, "max": 200, "step": 1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0, "step": 0.01}),
                "beams": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
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
        pImge = pImge.to('cuda:0')

        # Tokenize the prompt
        prompt = tokenizer.encode(prompt, return_tensors='pt', padding=False, truncation=False, add_special_tokens=False)
        # Embed image
        with torch.amp.autocast_mode.autocast('cuda', enabled=True):
            vision_outputs = clip_model(pixel_values=pImge, output_hidden_states=True)
            image_features = vision_outputs.hidden_states[-2]
            embedded_images = image_adapter(image_features)
            embedded_images = embedded_images.to('cuda:0')

        # Embed prompt
        prompt_embeds = text_model.model.embed_tokens(prompt.to('cuda:0'))
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
        ], dim=1).to('cuda:0')
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
