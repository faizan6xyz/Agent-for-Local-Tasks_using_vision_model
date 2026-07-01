import os
import gc
import torch
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from qwen_vl_utils import process_vision_info
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
print(f"Loading {model_name}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)
processor = AutoProcessor.from_pretrained(
    model_name,
    min_pixels=256 * 28 * 28,
    max_pixels=512 * 28 * 28
)

print("Model loaded successfully.")

def describe_image(image_path, Botton , prompt: str = None, max_new_tokens: int = 300):
    if prompt is None:
        prompt = f"This is a 1920x1080 desktop screenshot. Identify the {Botton} s. Return ONLY the bounding box coordinates [x_min, y_min, x_max, y_max] in JSON format."
    torch.cuda.empty_cache()
    gc.collect()
    img = Image.open(image_path).convert("RGB")
    target_size = 640
    img.thumbnail((target_size, target_size), Image.LANCZOS)
    
    folder_path = os.path.dirname(image_path)
    resized_path = os.path.join(folder_path, "temp_resized.jpg")
    img.save(resized_path, quality=85) 
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": resized_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
            response = processor.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
    finally:
        if os.path.exists(resized_path):
            os.remove(resized_path)
        del inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()
    return response
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    image_file = "data/screenshot_2026-07-01_16-17-06.png"
    
    if os.path.exists(image_file):
        print("Analyzing image...")
        result = describe_image(image_file)
        print("-" * 20)
        print("LLM Response:")
        print(result)
        print("-" * 20)
    else:
        print(f"Error: {image_file} not found. Please take a screenshot first.")