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
        mask: Path | None = Input(description="Image of the mask (black bg, white mask).", default=None),
        prompt: str = Input(description="Describe the model you would like to generate.", default="a photography of a model"),
        negative_prompt: str = Input(default=""),
        enable_cloth_guidance: bool = Input(description="Whether to enable cloth guidance or not.", default=False),
        seed: int = Input(default=-1),
        guidance_scale: float = Input(default=2.5),
        cloth_guidance_scale: float = Input(default=2.5),
        steps: int = Input(default=20),
        height: int = Input(default=768),
        width: int = Input(default=576),
    ) -> Path:
        """Run a single prediction on the model"""
        device = "cuda"

        model_path = "./checkpoints/magic_clothing_768_vitonhd_joint.safetensors"
        pipe_path = "SG161222/Realistic_Vision_V4.0_noVAE"

        print("Loading Cloth Image")
        cloth_image = Image.open(image).convert("RGB")

        if mask: print("Loading Mask Image")
        mask_image = Image.open(mask).convert("RGB") if mask else None

        print("Loading VAE")
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dtype=torch.float16)

        print("Loading Diffusion Pipeline")

        if enable_cloth_guidance:
            pipe = OmsDiffusionPipeline.from_pretrained(pipe_path, vae=vae, torch_dtype=torch.float16)
            print("Loaded OmsDiffusionPipeline")
        else:
            pipe = StableDiffusionPipeline.from_pretrained(pipe_path, vae=vae, torch_dtype=torch.float16)
            print("Loaded StableDiffusionPipeline")
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        print("Loading Cloth Adapter")

        full_net = ClothAdapter(pipe, model_path, device, enable_cloth_guidance, False)

        print("Generating Image")
        images, _ = full_net.generate(cloth_image, mask_image, prompt, "best quality, high quality", 1, negative_prompt, seed, guidance_scale, cloth_guidance_scale, steps, height, width)

        print("Saving Image")
        out_path = Path(tempfile.mkdtemp()) / "out.png"
        images[0].save(str(out_path))

        print("Done!")
        return out_path
