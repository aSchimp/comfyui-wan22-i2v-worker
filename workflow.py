"""Fixed native ComfyUI workflow construction."""
from __future__ import annotations

import os
from pathlib import Path
import re


MODEL_BASENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}\.safetensors$")


def safe_basename(value: str) -> str:
    if not isinstance(value, str) or Path(value).name != value or not MODEL_BASENAME.fullmatch(value):
        raise ValueError("invalid runtime configuration")
    return value


def model_config(env=os.environ) -> dict[str, str]:
    return {key: safe_basename(env.get(key, "")) for key in
            ("HIGH_MODEL", "LOW_MODEL", "TEXT_ENCODER", "VAE_MODEL")}


def dimensions(width: int, height: int, budget: int = 640 * 640) -> tuple[int, int]:
    """Return aspect-preserving, multiples-of-16 dimensions near the budget."""
    if width < 1 or height < 1:
        raise ValueError("invalid image dimensions")
    scale = (budget / (width * height)) ** .5
    scale = min(scale, 1280 / width, 1280 / height)
    w = max(16, round(width * scale / 16) * 16)
    h = max(16, round(height * scale / 16) * 16)
    return w, h


def build_workflow(*, input_name: str, output_prefix: str, prompt: str,
                   negative_prompt: str, seed: int, duration: int,
                   width: int, height: int, loras: list[dict], config: dict[str, str]) -> dict:
    """Create the only accepted graph; request data never supplies node types or paths."""
    w, h = dimensions(width, height)
    frames, midpoint = duration * 16 + 1, 3
    graph = {
        "1": {"class_type": "LoadImage", "inputs": {"image": input_name}},
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": config["HIGH_MODEL"], "weight_dtype": "fp8_e4m3fn"}},
        "3": {"class_type": "UNETLoader", "inputs": {"unet_name": config["LOW_MODEL"], "weight_dtype": "fp8_e4m3fn"}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": config["TEXT_ENCODER"], "type": "wan", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": config["VAE_MODEL"]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["4", 0]}},
        "8": {"class_type": "WanImageToVideo", "inputs": {"positive": ["6", 0], "negative": ["7", 0], "vae": ["5", 0], "width": w, "height": h, "length": frames, "batch_size": 1, "start_image": ["1", 0]}},
    }
    high, low = "2", "3"
    for lora in loras:
        if lora["high_strength"] > 0:
            node = str(10 + len(graph)); graph[node] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": [high, 0], "lora_name": lora["high"], "strength_model": lora["high_strength"]}}; high = node
        if lora["low_strength"] > 0:
            node = str(10 + len(graph)); graph[node] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": [low, 0], "lora_name": lora["low"], "strength_model": lora["low_strength"]}}; low = node
    graph.update({
        "9": {"class_type": "ModelSamplingSD3", "inputs": {"model": [high, 0], "shift": 7}},
        "10": {"class_type": "ModelSamplingSD3", "inputs": {"model": [low, 0], "shift": 7}},
        "90": {"class_type": "KSamplerAdvanced", "inputs": {"model": ["9", 0], "add_noise": "enable", "noise_seed": seed, "steps": 6, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "positive": ["8", 0], "negative": ["8", 1], "latent_image": ["8", 2], "start_at_step": 0, "end_at_step": midpoint, "return_with_leftover_noise": "enable"}},
        "91": {"class_type": "KSamplerAdvanced", "inputs": {"model": ["10", 0], "add_noise": "disable", "noise_seed": seed, "steps": 6, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "positive": ["8", 0], "negative": ["8", 1], "latent_image": ["90", 0], "start_at_step": midpoint, "end_at_step": 6, "return_with_leftover_noise": "disable"}},
        "92": {"class_type": "VAEDecode", "inputs": {"samples": ["91", 0], "vae": ["5", 0]}},
        "93": {"class_type": "CreateVideo", "inputs": {"images": ["92", 0], "fps": 16}},
        "94": {"class_type": "SaveVideo", "inputs": {"video": ["93", 0], "filename_prefix": output_prefix, "format": "mp4", "codec": "h264", "codec.encoding": "re-encode", "codec.encoding.crf": 28}},
    })
    return graph
