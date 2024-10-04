.2024-9-21 Joycaption Alpha Two - [Release article](https://civitai.com/articles/7697/joycaption-alpha-two-release)

Use instruct with alpha two, use non-instruct with alpha one.

```
CAPTION_TYPE_MAP = {
        "Descriptive": [
            "Write a descriptive caption for this image in a formal tone.",
            "Write a descriptive caption for this image in a formal tone within {word_count} words.",
            "Write a {length} descriptive caption for this image in a formal tone.",
        ],
        "Descriptive (Informal)": [
            "Write a descriptive caption for this image in a casual tone.",
            "Write a descriptive caption for this image in a casual tone within {word_count} words.",
            "Write a {length} descriptive caption for this image in a casual tone.",
        ],
        "Training Prompt": [
            "Write a stable diffusion prompt for this image.",
            "Write a stable diffusion prompt for this image within {word_count} words.",
            "Write a {length} stable diffusion prompt for this image.",
        ],
        "MidJourney": [
            "Write a MidJourney prompt for this image.",
            "Write a MidJourney prompt for this image within {word_count} words.",
            "Write a {length} MidJourney prompt for this image.",
        ],
        "Booru tag list": [
            "Write a list of Booru tags for this image.",
            "Write a list of Booru tags for this image within {word_count} words.",
            "Write a {length} list of Booru tags for this image.",
        ],
        "Booru-like tag list": [
            "Write a list of Booru-like tags for this image.",
            "Write a list of Booru-like tags for this image within {word_count} words.",
            "Write a {length} list of Booru-like tags for this image.",
        ],
        "Art Critic": [
            "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
            "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
            "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
        ],
        "Product Listing": [
            "Write a caption for this image as though it were a product listing.",
            "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
            "Write a {length} caption for this image as though it were a product listing.",
        ],
        "Social Media Post": [
            "Write a caption for this image as if it were being used for a social media post.",
            "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
            "Write a {length} caption for this image as if it were being used for a social media post.",
        ],
    }
```

.2024-9-21 Joycaption Alpha One - [Release thread](https://www.reddit.com/r/StableDiffusion/comments/1fm9pxa/joycaption_free_open_uncensored_vlm_alpha_one/)

Below is how the space maps the different options to prompts, use these prompts in the node.
```
CAPTION_TYPE_MAP = {
	("descriptive", "formal", False, False): ["Write a descriptive caption for this image in a formal tone."],
	("descriptive", "formal", False, True): ["Write a descriptive caption for this image in a formal tone within {word_count} words."],
	("descriptive", "formal", True, False): ["Write a {length} descriptive caption for this image in a formal tone."],
	("descriptive", "informal", False, False): ["Write a descriptive caption for this image in a casual tone."],
	("descriptive", "informal", False, True): ["Write a descriptive caption for this image in a casual tone within {word_count} words."],
	("descriptive", "informal", True, False): ["Write a {length} descriptive caption for this image in a casual tone."],

	("training_prompt", "formal", False, False): ["Write a stable diffusion prompt for this image."],
	("training_prompt", "formal", False, True): ["Write a stable diffusion prompt for this image within {word_count} words."],
	("training_prompt", "formal", True, False): ["Write a {length} stable diffusion prompt for this image."],

	("rng-tags", "formal", False, False): ["Write a list of Booru tags for this image."],
	("rng-tags", "formal", False, True): ["Write a list of Booru tags for this image within {word_count} words."],
	("rng-tags", "formal", True, False): ["Write a {length} list of Booru tags for this image."],
}
```

.2024-9-9 florence2 Add Florence-2-large-PromptGen-v1.5 and MiniCPM3-4B(CXH_MinCP3_4B_Load CXH_MinCP3_4B_Chat) 
    MiniCPM3-4B聊天 翻译，改写都很强

.2024-9-6 florence2 Add Florence-2-base-PromptGen-v1.5 

.2024-9-2 更新批量打标案例(Update batch marking cases) 速度：florence2<min2.6<joy

![1724901350282](https://github.com/user-attachments/assets/c9d9cd10-fbd6-4aeb-91b6-f2740c3998cc)

(1).基于comfyui节点图片放推(Recommended based on comfyui node pictures)

    1.Joy_caption

    2.miniCPMv2_6_prompt_generator

    3.florence2

(2).安装(Installation)：

  1.（Comfyui evn python.exe） python -m pip install -r requirements.txt or click install_req.bat

  注意：transformers 版本不能太低（Note: The version of transformers cannot be too low）

  2. 下载模型或者运行comfyui自动下载模型到合适文件夹(Download the model or run Comfyui to automatically download the model to the appropriate folder)

(3) 模型安装（Install model）

   1).Joy_caption

   .运行自动下载模型(推荐手动下载) Run automatic download model (manual download recommended)
   
    1.https://huggingface.co/google/siglip-so400m-patch14-384 放到(put in)clip/siglip-so400m-patch14-384
      
![1724901434148](https://github.com/user-attachments/assets/12ad9627-e121-4bc8-98cc-313fa491bde4)

    
    2. https://huggingface.co/unsloth/Meta-Llama-3.1-8B-bnb-4bit 放到(put in)LLM/Meta-Llama-3.1-8B-bnb-4bit
      
![1724901495135](https://github.com/user-attachments/assets/3cac31a7-8150-4d78-96d1-8aa3198fe572)


    3.必须手动下载(Must be downloaded manually):https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha/tree/main/wpkklhc6   (put in)Joy_caption 

![1724901527482](https://github.com/user-attachments/assets/e8ec1be6-a96c-4e73-9422-7bcdafb8f1d4)

    4.必须手动下载(Must be downloaded manually):https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-one/tree/main/9em124t2-499968   (put in)Joy_caption_alpha_one 

![1724901527482](https://github.com/user-attachments/assets/e8ec1be6-a96c-4e73-9422-7bcdafb8f1d4)

    5.必须手动下载(Must be downloaded manually):https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two/tree/main/cgrkzexw-599808   (put in)Joy_caption_alpha_two

![image](https://github.com/user-attachments/assets/8a7937dd-d58e-46ef-90bd-5213ab970a3c)


 2).MiniCPMv2_6-prompt-generator + CogFlorence
 
 https://huggingface.co/pzc163/MiniCPMv2_6-prompt-generator
 
 https://huggingface.co/thwri/CogFlorence-2.2-Large
 
 ![1724902196890](https://github.com/user-attachments/assets/22373c22-8083-4b3f-af10-774d86560f16)

 Run with:flux1-dev-Q8_0.gguf

 ![e8ad7fa14f807184a99ea23b31e8a60](https://github.com/user-attachments/assets/178ee440-919e-4b28-b1bd-c2c1e2e0ceb4)

 ![1724897220972](https://github.com/user-attachments/assets/ac3c072d-dccc-4f29-bcbd-45c7945407be)

 ![1724897584034](https://github.com/user-attachments/assets/584adc69-3e0d-4cb9-8392-0fe337dc34a2)








