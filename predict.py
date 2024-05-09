from cog import BasePredictor, Input, Path
import os.path
import pdb

import torch
from diffusers import UniPCMultistepScheduler, AutoencoderKL
from diffusers.pipelines import StableDiffusionPipeline
from PIL import Image
import argparse

from garment_adapter.garment_diffusion import ClothAdapter
from pipelines.OmsDiffusionPipeline import OmsDiffusionPipeline


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./checkpoints/magic_clothing_768_vitonhd_joint.safetensors")

    def predict(
        self,
        image: Path = Input(description="Grayscale input image")
    ) -> Path:
        """Run a single prediction on the model"""
        device = "cuda"

        args = {
            "model_path": "./checkpoints/magic_clothing_768_vitonhd_joint.safetensors",
            "pipe_path": "SG161222/Realistic_Vision_V4.0_noVAE",
            "cloth_path": image,
            "enable_cloth_guidance": True,
        }

        cloth_image = Image.open(args['cloth_path']).convert("RGB")

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dtype=torch.float16)
        if args['enable_cloth_guidance']:
            pipe = OmsDiffusionPipeline.from_pretrained(args['pipe_path'], vae=vae, torch_dtype=torch.float16)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(args['pipe_path'], vae=vae, torch_dtype=torch.float16)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        full_net = ClothAdapter(pipe, args['model_path'], device, args['enable_cloth_guidance'], False)
        images = full_net.generate(cloth_image)

        return images[0]
