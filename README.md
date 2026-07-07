# Agent for Local Tasks using Vision Model

A local, GPU-accelerated desktop automation agent that uses **Qwen2.5-VL-3B-Instruct** to *see* the screen, reason about the next action, and control the mouse/keyboard to accomplish a natural-language goal — entirely offline, with no cloud API calls.

Give it a goal like `"open notepad and type hello world"`, and the agent will:
1. Take a screenshot of the current desktop
2. Feed it to a local vision-language model along with the goal
3. Parse the model's suggested next action (click, type, scroll, etc.)
4. Execute that action with `pyautogui`
5. Repeat until the model reports the goal is done (or a step limit / stuck-loop is hit)

---

## How it works

```
┌─────────────┐     ┌───────────────────┐     ┌────────────────────┐     ┌───────────────┐
│ Screenshot   │ --> │ Qwen2.5-VL (4-bit) │ --> │  Parse JSON action  │ --> │ pyautogui exec │
│ (snapshot.py)│     │ reasons on image + │     │ {action, coords,   │     │ (Mouse_use.py) │
└─────────────┘     │ goal, returns JSON │     │  text_input, ...}  │     └───────┬────────┘
                     └───────────────────┘     └────────────────────┘             │
                                ▲                                                 │
                                └─────────────── loop until "done" ───────────────┘
```

The model is prompted to return **strict JSON** describing exactly one action at a time (not a full plan), which keeps it grounded in the *current* screen state rather than hallucinating multi-step plans.

---

## Repository structure

| File | Purpose |
|---|---|
| `optimized_qwen_vision.py` | **Main entry point.** Full agent loop — loads the model once, takes a screenshot each step, asks the VLM for the next action, executes it, and repeats until done or `max_steps` is reached. |
| `Qwen_vision.py` | Simpler/earlier standalone script for describing a single image with the VLM (used for prototyping / one-off vision queries, e.g. locating a specific UI element in a screenshot). |
| `snapshot.py` | Captures a full-desktop screenshot via `PIL.ImageGrab` and saves it to `data/` with a timestamped filename. |
| `Mouse_use.py` | Thin wrapper around `pyautogui` for mouse/keyboard actions: `move_to`, `click_at`, `double_click_at`, `right_click_at`, `drag_to`, `type_text`, `wait`. |

---

## Requirements

- Python 3.10+
- An NVIDIA GPU with CUDA support (tested on 4GB VRAM using 4-bit quantization)
- Windows/Linux desktop environment (uses `pyautogui`, so a GUI session is required — won't work headless)

### Install dependencies

```bash
pip install torch transformers accelerate bitsandbytes pyautogui pillow qwen-vl-utils
```

> **Note:** `bitsandbytes` 4-bit quantization support on Windows can be finicky — if you hit issues, check for a Windows-compatible build or run inside WSL2 with CUDA passthrough.

---

## Usage

Run the main agent:

```bash
python optimized_qwen_vision.py
```

You'll be prompted for a goal:

```
enter your goal : open the start menu
```

The agent will then loop, screenshotting and acting, printing each step's reasoning:

```
Screenshot: data/screenshot_2026-07-07_04-15-45.png
 -> action=click target='Windows Start button' coords=(1546,1053) reasoning=Start menu icon is visible in the taskbar
 step took 2.14s
```

### Standalone image description (`Qwen_vision.py`)

For quick one-off tests without running the full action loop:

```bash
python Qwen_vision.py
```

This loads the model and runs a single description/localization prompt against a hardcoded image path — useful for debugging what the VLM "sees" before wiring it into the full agent.

---

## Key design details

- **4-bit quantization (`bitsandbytes` NF4)** — lets Qwen2.5-VL-3B run on consumer GPUs with as little as 4GB VRAM.
- **Dynamic resolution handling** — the model is told the true screen resolution (`pyautogui.size()`) so it can return coordinates already scaled to the real desktop, avoiding manual rescaling logic.
- **JSON-only responses** — the system prompt forces the model to reply with a single structured JSON action, parsed defensively (`parse_action_json`) to strip stray markdown/code fences the model sometimes adds.
- **Stuck-loop detection** — if the model proposes the identical action repeatedly, the agent breaks out early instead of looping forever.
- **Periodic memory cleanup** — `torch.cuda.empty_cache()` / `gc.collect()` run every few steps rather than every step, since these calls force expensive CUDA synchronization.

---

## Known limitations

- Single-step reasoning only — the model has no memory of prior steps beyond the current screenshot, so it can occasionally repeat or contradict earlier actions.
- Coordinate accuracy depends heavily on the VLM's grounding ability at 3B scale; larger models (7B+) will generally localize UI elements more precisely.
- No sandboxing — the agent directly controls your real mouse/keyboard, so run it in a context where you're comfortable with an AI clicking around your desktop.

---

## Roadmap ideas

- [ ] Add short-term memory of prior N actions/screenshots for better multi-step coherence
- [ ] Swap in a larger/quantized VLM (e.g. Qwen2.5-VL-7B) as an optional backend
- [ ] Add a "confirm before executing" safety mode for destructive actions
- [ ] Package as a CLI with configurable model/backend (local vs. NIM-hosted)
