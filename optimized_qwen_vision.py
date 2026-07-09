import json
import os
import gc
import re
import time
import torch
import pyautogui
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info
from Mouse_use import move_to, click_at, right_click_at, double_click_at, drag_to, type_text, wait
from snapshot import take_full_screenshot
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
RESIZE_TARGET = 640
MAX_NEW_TOKENS = 200          # JSON replies are short; 300 was more than needed
CLEANUP_EVERY_N_STEPS = 3     # empty_cache()/gc.collect() are expensive syncs - don't do them every step
LOOP_BREAK_THRESHOLD = 2      # stop early if the model repeats the identical action N times in a row
print(f"Loading {MODEL_NAME}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,     
    attn_implementation="sdpa",)
model.eval()
model.generation_config.do_sample = False   
model.generation_config.num_beams = 1
model.generation_config.use_cache = True
processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    min_pixels=256 * 28 * 28,
    max_pixels=512 * 28 * 28,)
print("Model loaded successfully.")
SCREEN_W, SCREEN_H = pyautogui.size()
print(f"Detected screen resolution: {SCREEN_W}x{SCREEN_H}")
def create_vlm_prompt(goal, screen_w, screen_h):
    system_instruction = f"""You are an expert Browser/Desktop Automation Agent.
Analyze the provided screenshot and determine the single best next action to achieve the user's goal.

Rules:
1. The screenshot corresponds to a desktop of resolution {screen_w}x{screen_h}. All coordinates you
   return MUST be scaled to that resolution, based on the position of the element within the image.
2. Identify interactive elements: buttons, links, input fields, checkboxes.
3. If the goal is already achieved, return action "done".
4. If you need to type, specify the text clearly in text_input.
5. Take ONE step at a time. Do not assume future steps.
6. Return ONLY valid JSON. No markdown, no code fences, no extra text."""
    user_prompt = f"""GOAL: "{goal}"

Analyze the current state of the desktop in the screenshot.
What is the single specific next step to move closer to the goal?

Respond in exactly this JSON format:
{{
    "action": "click" | "move_to" | "drag_to" | "double_click" | "right_click" | "type" | "scroll" | "wait" | "done",
    "target_element": "Brief description of the element (e.g., 'Blue Search Button')",
    "coordinates": {{"x": <int>, "y": <int>}},
    "text_input": "<string if action is type, else null>",
    "reasoning": "Why this action is necessary now."
}}"""
    return system_instruction, user_prompt
def get_next_action(goal, image_path, max_new_tokens=MAX_NEW_TOKENS):
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((RESIZE_TARGET, RESIZE_TARGET), Image.LANCZOS)

    system_instruction, user_prompt = create_vlm_prompt(goal, SCREEN_W, SCREEN_H)

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": user_prompt},],},]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",).to(model.device)
    with torch.inference_mode():  
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = processor.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,)
    del inputs, outputs
    action_dict = parse_action_json(response)
    return action_dict, response
def maybe_cleanup(step):
    if step % CLEANUP_EVERY_N_STEPS == 0:
        torch.cuda.empty_cache()
        gc.collect()
def parse_action_json(raw_response):
    cleaned = raw_response.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"Could not parse model response as JSON:\n{raw_response}")
def execute_action(action_dict):
    action = action_dict.get("action")
    coords = action_dict.get("coordinates") or {}
    x, y = coords.get("x"), coords.get("y")
    text_input = action_dict.get("text_input")
    print(f"  -> action={action} target='{action_dict.get('target_element')}' "
          f"coords=({x},{y}) reasoning={action_dict.get('reasoning')}")
    if action == "click":
        click_at(x, y)
    elif action == "double_click":
        double_click_at(x, y)
    elif action == "right_click":
        right_click_at(x, y)
    elif action == "move_to":
        move_to(x, y)
    elif action == "drag_to":
        drag_to(x, y)
    elif action == "type":
        if not text_input:
            print("  Warning: action was 'type' but text_input was empty/null. Skipping.")
        else:
            type_text(text_input)
    elif action == "scroll":
        pyautogui.scroll(-500)
    elif action == "wait":
        wait(1)
    elif action == "done":
        print("Goal achieved.")
        return True
    else:
        print(f"  Warning: unknown action '{action}', skipping.")
    return False
def _action_signature(action_dict):
    coords = action_dict.get("coordinates") or {}
    return (
        action_dict.get("action"),
        coords.get("x"),
        coords.get("y"),
        action_dict.get("text_input"),
    )
def run_agent(goal, max_steps=15, delay_between_steps=1.5):
    os.makedirs("data", exist_ok=True)
    last_signature = None
    repeat_count = 0
    for step in range(1, max_steps + 1):
        t0 = time.time()
        screenshot_path = take_full_screenshot()
        print(f"Screenshot: {screenshot_path}")
        try:
            action_dict, raw_response = get_next_action(goal, screenshot_path)
        except ValueError as e:
            print(f"  Failed to parse model response, retrying next step. {e}")
            maybe_cleanup(step)
            time.sleep(delay_between_steps)
            continue
        signature = _action_signature(action_dict)
        if signature == last_signature:
            repeat_count += 1
            if repeat_count >= LOOP_BREAK_THRESHOLD:
                print(f"\nDetected repeated identical action {repeat_count + 1}x in a row - "
                      f"stopping to avoid a stuck loop.")
                return
        else:
            repeat_count = 0
        last_signature = signature
        is_done = execute_action(action_dict)
        print(f"  step took {time.time() - t0:.2f}s")
        maybe_cleanup(step)
        if is_done:
            print(f"\nAgent finished goal in {step} step(s).")
            return
        time.sleep(delay_between_steps)
    print(f"\nReached max_steps={max_steps} without the model reporting 'done'.")
if __name__ == "__main__":
    goal = input("enter your goal : ")
    run_agent(goal, max_steps=10)