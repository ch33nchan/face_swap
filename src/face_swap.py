import os
from typing import Optional, Union, List
import torch
import numpy as np
from PIL import Image
import cv2
from insightface.app import FaceAnalysis
from diffusers import FluxPipeline, AutoencoderKL
from diffusers.models.attention_processor import AttnProcessor2_0
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceSwapPipeline:
    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.1-dev",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16,
        lora_path: Optional[str] = None,
        lora_scale: float = 1.0,
    ):
        self.device = device
        self.dtype = dtype
        
        logger.info(f"Initializing face analysis on {device}")
        self.face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        )
        self.face_app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))
        
        hf_token = os.getenv("HF_TOKEN")
        
        logger.info(f"Loading FLUX pipeline: {model_id}")
        self.pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            token=hf_token,
        ).to(device)
        
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        
        if lora_path and os.path.exists(lora_path):
            logger.info(f"Loading LORA from {lora_path}")
            self.pipe.load_lora_weights(lora_path)
            self.pipe.fuse_lora(lora_scale=lora_scale)
        
        logger.info("Pipeline initialized successfully")

    def extract_face_embedding(self, image: Union[str, np.ndarray, Image.Image]) -> Optional[np.ndarray]:
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image
        
        faces = self.face_app.get(img)
        if len(faces) == 0:
            logger.warning("No faces detected in reference image")
            return None
        
        return faces[0].embedding

    def prepare_reference_conditioning(
        self,
        reference_image: Union[str, Image.Image],
        prompt: str = "",
    ) -> dict:
        if isinstance(reference_image, str):
            ref_img = Image.open(reference_image).convert("RGB")
        else:
            ref_img = reference_image
        
        face_embedding = self.extract_face_embedding(ref_img)
        if face_embedding is None:
            raise ValueError("No face detected in reference image")
        
        return {
            "reference_image": ref_img,
            "face_embedding": face_embedding,
            "prompt": prompt
        }

    def swap_face(
        self,
        base_image: Union[str, Image.Image],
        reference_image: Union[str, Image.Image],
        prompt: str = "high quality portrait, detailed face, natural lighting",
        negative_prompt: str = "blurry, low quality, distorted face, artifacts",
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        strength: float = 0.75,
        seed: Optional[int] = None,
    ) -> Image.Image:
        
        if isinstance(base_image, str):
            base_img = Image.open(base_image).convert("RGB")
        else:
            base_img = base_image
        
        ref_cond = self.prepare_reference_conditioning(reference_image, prompt)
        
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)
        
        logger.info("Running face swap inference")
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=base_img,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        
        return result

    def batch_swap(
        self,
        base_images: List[Union[str, Image.Image]],
        reference_image: Union[str, Image.Image],
        **kwargs
    ) -> List[Image.Image]:
        results = []
        for base_img in base_images:
            try:
                result = self.swap_face(base_img, reference_image, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process image: {e}")
                results.append(None)
        return results

    def unload(self):
        del self.pipe
        del self.face_app
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
