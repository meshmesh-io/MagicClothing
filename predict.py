from cog import BasePredictor, Input, Path

import tempfile
import torch
from diffusers import UniPCMultistepScheduler, AutoencoderKL
from diffusers.pipelines import StableDiffusionPipeline
from PIL import Image

from garment_adapter.garment_diffusion import ClothAdapter
from pipelines.OmsDiffusionPipeline import OmsDiffusionPipeline


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./checkpoints/magic_clothing_768_vitonhd_joint.safetensors")

    def predict(
        self,
        image: Path = Input(description="Image of the shirt to be worn over the person's body."),
        enable_cloth_guidance: bool = Input(description="Whether to enable cloth guidance or not.", default=False),
        prompt: str = Input(description="Describe the model you would like to generate.", default="a photography of a model"),
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
        images, _ = full_net.generate(cloth_image, None, prompt, "best quality, high quality", 1, None, -1, 7.5, 2.5, 20, 576, 768)

        print("Saving Image")
        out_path = Path(tempfile.mkdtemp() / "out.png")
        images[0].save(str(out_path))

        print("Done!")
        return out_path
