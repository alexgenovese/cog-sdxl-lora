#!/usr/bin/env python
# Run this before you deploy it on replicate, because if you don't
# whenever you run the model, it will download the weights from the
# internet, which will take a long time.
import os
import torch
from diffusers import AutoencoderKL, DiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import ( StableDiffusionSafetyChecker )
from transformers import (Blip2Processor, CLIPSegProcessor, Swin2SRForImageSuperResolution)

# Define the correct VAE
better_vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

# Check folder exists
if not os.path.exists("./refiner-cache"):
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=better_vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    pipe.save_pretrained("./sdxl-cache", safe_serialization=True)

# Check folder exists
if not os.path.exists("./refiner-cache"):
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    # TODO - we don't need to save all of this and in fact should save just the unet, tokenizer, and config.
    pipe.save_pretrained("./refiner-cache", safe_serialization=True)

# Check folder exists
if not os.path.exists("./safety-cache"):
    safety = StableDiffusionSafetyChecker.from_pretrained(
        "CompVis/stable-diffusion-safety-checker",
        torch_dtype=torch.float16,
    )
    safety.save_pretrained("./safety-cache")

# Download the preprocess models
BLIP2_REPO_ID = "Salesforce/blip2-opt-2.7b"
BLIP_2_FOLDER = "/Blip2"
CIDAS_REPO_ID = "CIDAS/clipseg-rd64-refined"
CIDAS_FOLDER = "/CIDAS"
UPSCALER_REPO_ID = "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"
UPSCALER_FOLDER = "/Caidas"
PREPROCESS_MODEL_CACHE = "./preprocess-cache"

# Check folder exists
if not os.path.exists(PREPROCESS_MODEL_CACHE+UPSCALER_FOLDER):
    upscaler = Swin2SRForImageSuperResolution.from_pretrained(UPSCALER_REPO_ID, cache_dir=PREPROCESS_MODEL_CACHE)
    upscaler.save_pretrained(PREPROCESS_MODEL_CACHE+UPSCALER_FOLDER, safe_serialization=True)

if not os.path.exists(PREPROCESS_MODEL_CACHE+CIDAS_REPO_ID):
    cidas = CLIPSegProcessor.from_pretrained(CIDAS_REPO_ID, cache_dir=PREPROCESS_MODEL_CACHE)
    cidas.save_pretrained(PREPROCESS_MODEL_CACHE+CIDAS_FOLDER, safe_serialization=True)

if not os.path.exists(PREPROCESS_MODEL_CACHE+BLIP_2_FOLDER):
    blip2 = Blip2Processor.from_pretrained(BLIP2_REPO_ID, cache_dir=PREPROCESS_MODEL_CACHE)
    blip2.save_pretrained(PREPROCESS_MODEL_CACHE+BLIP_2_FOLDER, safe_serialization=True)