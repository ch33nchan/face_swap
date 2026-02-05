#!/usr/bin/env python3

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import torch
from diffusers import FluxPipeline
from insightface.app import FaceAnalysis
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    csv_path: Path
    output_dir: Path
    fix_column: str
    fix_value: str
    ref_angle_col: str
    front_angle_col: str
    original_img_col: str
    swapped_img_col: str
    gpu_id: Optional[int]
    model_id: str
    lora_path: Optional[Path]
    guidance_scale: float
    num_inference_steps: int


class ImageDownloader:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
    
    def download(self, url: str) -> Path:
        parsed = urlparse(url)
        filename = Path(parsed.path).name
        local_path = self.cache_dir / filename
        
        if local_path.exists():
            logger.debug(f"Using cached: {filename}")
            return local_path
        
        logger.info(f"Downloading: {url}")
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        
        local_path.write_bytes(response.content)
        return local_path


class FaceSwapProcessor:
    def __init__(self, device: str):
        self.device = device
        logger.info(f"Initializing FaceAnalysis on {device}")
        self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
    
    def extract_face_embedding(self, image_path: Path) -> Optional[np.ndarray]:
        img = np.array(Image.open(image_path).convert('RGB'))
        faces = self.app.get(img)
        if not faces:
            logger.warning(f"No face detected in {image_path}")
            return None
        return faces[0].embedding
    
    def swap_face(self, source_path: Path, target_path: Path, output_path: Path):
        source_img = np.array(Image.open(source_path).convert('RGB'))
        target_img = np.array(Image.open(target_path).convert('RGB'))
        
        source_faces = self.app.get(source_img)
        target_faces = self.app.get(target_img)
        
        if not source_faces or not target_faces:
            logger.error("Face detection failed")
            return None
        
        result = self.app.get(source_img, target_faces[0])
        Image.fromarray(result).save(output_path)
        return output_path


class FluxInference:
    def __init__(self, model_id: str, device: str, lora_path: Optional[Path] = None):
        self.device = device
        logger.info(f"Loading Flux model: {model_id}")
        
        self.pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16
        ).to(device)
        
        if lora_path and lora_path.exists():
            logger.info(f"Loading LORA: {lora_path}")
            self.pipe.load_lora_weights(str(lora_path))
    
    def generate(
        self,
        prompt: str,
        ref_image: Image.Image,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 50,
        height: int = 1024,
        width: int = 1024
    ) -> Image.Image:
        with torch.inference_mode():
            image = self.pipe(
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                generator=torch.Generator(device=self.device).manual_seed(42)
            ).images[0]
        return image


class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        self.downloader = ImageDownloader(config.output_dir / 'cache')
        
        device = f'cuda:{config.gpu_id}' if config.gpu_id is not None else 'cuda'
        
        logger.info("Initializing Flux pipeline")
        self.flux = FluxInference(
            config.model_id,
            device,
            config.lora_path
        )
        
        logger.info("Initializing FaceSwap")
        self.face_swap = FaceSwapProcessor(device)
        
        self.results_dir = config.output_dir / 'results'
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading CSV: {self.config.csv_path}")
        df = pd.read_csv(self.config.csv_path)
        logger.info(f"Total rows: {len(df)}")
        
        filter_mask = df[self.config.fix_column].str.lower() == self.config.fix_value.lower()
        filtered_df = df[filter_mask].copy()
        logger.info(f"Filtered rows ({self.config.fix_column}={self.config.fix_value}): {len(filtered_df)}")
        
        return filtered_df
    
    def process_row(self, row: pd.Series, index: int) -> Optional[str]:
        try:
            logger.info(f"Processing row {index}")
            
            original_path = self.downloader.download(row[self.config.original_img_col])
            swapped_path = self.downloader.download(row[self.config.swapped_img_col])
            ref_angle_path = self.downloader.download(row[self.config.ref_angle_col])
            front_angle_path = self.downloader.download(row[self.config.front_angle_col])
            
            original_img = Image.open(original_path).convert('RGB')
            ref_img = Image.open(ref_angle_path).convert('RGB')
            
            prompt = f"high quality portrait, character consistency, same face and features as reference"
            
            generated = self.flux.generate(
                prompt=prompt,
                ref_image=ref_img,
                guidance_scale=self.config.guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                height=original_img.height,
                width=original_img.width
            )
            
            temp_gen_path = self.results_dir / f"temp_gen_{index}.png"
            generated.save(temp_gen_path)
            
            output_path = self.results_dir / f"row_{index}_output.png"
            self.face_swap.swap_face(ref_angle_path, temp_gen_path, output_path)
            
            temp_gen_path.unlink()
            
            logger.info(f"Saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to process row {index}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run(self):
        df = self.load_data()
        
        results = []
        for idx, row in df.iterrows():
            output_path = self.process_row(row, idx)
            results.append({
                'index': idx,
                'output_path': output_path,
                'status': 'success' if output_path else 'failed'
            })
        
        results_df = pd.DataFrame(results)
        results_csv = self.config.output_dir / 'processing_results.csv'
        results_df.to_csv(results_csv, index=False)
        logger.info(f"Results saved: {results_csv}")
        
        success_count = results_df['status'].value_counts().get('success', 0)
        logger.info(f"Completed: {success_count}/{len(results_df)} successful")


def main():
    parser = argparse.ArgumentParser(description='Face Swap Character Consistency Pipeline')
    parser.add_argument('--csv', type=Path, required=True, help='Input CSV path')
    parser.add_argument('--output-dir', type=Path, required=True, help='Output directory')
    parser.add_argument('--fix-column', type=str, default='Fix Char Const', help='Filter column name')
    parser.add_argument('--fix-value', type=str, default='No', help='Filter value')
    parser.add_argument('--ref-angle-col', type=str, default='Reference Angle', help='Reference angle column')
    parser.add_argument('--front-angle-col', type=str, default='Front Angle', help='Front angle column')
    parser.add_argument('--original-img-col', type=str, default='Original Image', help='Original image column')
    parser.add_argument('--swapped-img-col', type=str, default='Swapped Image', help='Swapped image column')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--model-id', type=str, default='black-forest-labs/FLUX.1-dev', help='Flux model ID')
    parser.add_argument('--lora-path', type=Path, default=None, help='Path to LORA weights')
    parser.add_argument('--guidance-scale', type=float, default=3.5, help='Guidance scale')
    parser.add_argument('--num-inference-steps', type=int, default=50, help='Number of inference steps')
    
    args = parser.parse_args()
    
    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        logger.info(f"Set CUDA_VISIBLE_DEVICES={args.gpu_id}")
    
    config = Config(
        csv_path=args.csv,
        output_dir=args.output_dir,
        fix_column=args.fix_column,
        fix_value=args.fix_value,
        ref_angle_col=args.ref_angle_col,
        front_angle_col=args.front_angle_col,
        original_img_col=args.original_img_col,
        swapped_img_col=args.swapped_img_col,
        gpu_id=args.gpu_id,
        model_id=args.model_id,
        lora_path=args.lora_path,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps
    )
    
    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == '__main__':
    main()