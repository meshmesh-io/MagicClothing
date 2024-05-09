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
        image: Path = Input(description="Image of the shirt to be worn over the person's body."),
        enable_cloth_guidance: bool = Input(description="Whether to enable cloth guidance or not.")
    ) -> Path:
        """Run a single prediction on the model"""
        device = "cuda"

        args = {
            "model_path": "./checkpoints/magic_clothing_768_vitonhd_joint.safetensors",
            "pipe_path": "SG161222/Realistic_Vision_V4.0_noVAE",
            "cloth_path": image,
            "enable_cloth_guidance": enable_cloth_guidance,
        }

        print('Predicting with the following arguments:')
        print(args)

        print("Loading Cloth Image")

        cloth_image = Image.open(args['cloth_path']).convert("RGB")

        print("Loading VAE")

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dtype=torch.float16)
        
        print("Loading Diffusion Pipeline")

        if args['enable_cloth_guidance']:
            pipe = OmsDiffusionPipeline.from_pretrained(args['pipe_path'], vae=vae, torch_dtype=torch.float16)
            print("Loaded OmsDiffusionPipeline")
        else:
            pipe = StableDiffusionPipeline.from_pretrained(args['pipe_path'], vae=vae, torch_dtype=torch.float16)
            print("Loaded StableDiffusionPipeline")
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        print("Loading Cloth Adapter")

        full_net = ClothAdapter(pipe, args['model_path'], device, args['enable_cloth_guidance'], False)

        print("Generating Image")
        images = full_net.generate(cloth_image)

        print("Saving Image")

        return images[0]
