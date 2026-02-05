#!/usr/bin/env python3

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import pandas as pd
import requests
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
    workflow_path: Path
    output_dir: Path
    comfyui_url: str
    fix_column: str
    fix_value: str
    ref_angle_col: str
    front_angle_col: str
    original_img_col: str
    swapped_img_col: str
    generated_img_col: str
    batch_size: int
    poll_interval: int
    timeout: int
    gpu_id: Optional[int]


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


class ComfyUIClient:
    def __init__(self, base_url: str, timeout: int = 300):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
    
    def upload_image(self, image_path: Path) -> str:
        url = f"{self.base_url}/upload/image"
        with open(image_path, 'rb') as f:
            files = {'image': (image_path.name, f, 'image/png')}
            response = self.session.post(url, files=files, timeout=30)
            response.raise_for_status()
            return response.json()['name']
    
    def queue_prompt(self, workflow: Dict) -> str:
        url = f"{self.base_url}/prompt"
        payload = {"prompt": workflow}
        response = self.session.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()['prompt_id']
    
    def get_history(self, prompt_id: str) -> Optional[Dict]:
        url = f"{self.base_url}/history/{prompt_id}"
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        history = response.json()
        return history.get(prompt_id)
    
    def wait_for_completion(self, prompt_id: str, poll_interval: int = 2) -> Dict:
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            history = self.get_history(prompt_id)
            if history and history.get('status', {}).get('completed', False):
                return history
            time.sleep(poll_interval)
        raise TimeoutError(f"Job {prompt_id} did not complete within {self.timeout}s")
    
    def get_output_images(self, history: Dict) -> List[str]:
        outputs = []
        for node_output in history.get('outputs', {}).values():
            if 'images' in node_output:
                for img in node_output['images']:
                    outputs.append(img['filename'])
        return outputs
    
    def download_output(self, filename: str, output_path: Path):
        url = f"{self.base_url}/view"
        params = {'filename': filename, 'type': 'output'}
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        output_path.write_bytes(response.content)


class WorkflowProcessor:
    def __init__(self, workflow_template: Dict):
        self.template = workflow_template
    
    def create_workflow(
        self,
        original_img: str,
        swapped_img: str,
        ref_angle_img: str,
        front_angle_img: str
    ) -> Dict:
        workflow = json.loads(json.dumps(self.template))
        
        load_image_map = {
            137: ref_angle_img,
            141: front_angle_img,
            128: original_img,
            121: swapped_img,
            151: original_img
        }
        
        for node in workflow['nodes']:
            node_id = node.get('id')
            if node.get('type') == 'LoadImage' and node_id in load_image_map:
                if 'widgets_values' in node:
                    node['widgets_values'][0] = load_image_map[node_id]
        
        return workflow


class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        self.downloader = ImageDownloader(config.output_dir / 'cache')
        self.client = ComfyUIClient(config.comfyui_url, config.timeout)
        
        with open(config.workflow_path) as f:
            workflow_template = json.load(f)
        self.processor = WorkflowProcessor(workflow_template)
        
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
            
            original_name = self.client.upload_image(original_path)
            swapped_name = self.client.upload_image(swapped_path)
            ref_angle_name = self.client.upload_image(ref_angle_path)
            front_angle_name = self.client.upload_image(front_angle_path)
            
            workflow = self.processor.create_workflow(
                original_name,
                swapped_name,
                ref_angle_name,
                front_angle_name
            )
            
            prompt_id = self.client.queue_prompt(workflow)
            logger.info(f"Queued job: {prompt_id}")
            
            history = self.client.wait_for_completion(prompt_id, self.config.poll_interval)
            output_images = self.client.get_output_images(history)
            
            if not output_images:
                logger.error(f"No output images for row {index}")
                return None
            
            output_filename = f"row_{index}_{output_images[0]}"
            output_path = self.results_dir / output_filename
            self.client.download_output(output_images[0], output_path)
            
            logger.info(f"Saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to process row {index}: {e}")
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
    parser = argparse.ArgumentParser(description='ComfyUI Character Consistency Pipeline')
    parser.add_argument('--csv', type=Path, required=True, help='Input CSV path')
    parser.add_argument('--workflow', type=Path, required=True, help='ComfyUI workflow JSON path')
    parser.add_argument('--output-dir', type=Path, required=True, help='Output directory')
    parser.add_argument('--comfyui-url', type=str, default='http://localhost:8188', help='ComfyUI API URL')
    parser.add_argument('--fix-column', type=str, default='Fix Char Const', help='Filter column name')
    parser.add_argument('--fix-value', type=str, default='No', help='Filter value')
    parser.add_argument('--ref-angle-col', type=str, default='Reference Angle', help='Reference angle column')
    parser.add_argument('--front-angle-col', type=str, default='Front Angle', help='Front angle column')
    parser.add_argument('--original-img-col', type=str, default='Original Image', help='Original image column')
    parser.add_argument('--swapped-img-col', type=str, default='Swapped Image', help='Swapped image column')
    parser.add_argument('--generated-img-col', type=str, default='Generated Image', help='Generated image column')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--poll-interval', type=int, default=2, help='Polling interval in seconds')
    parser.add_argument('--timeout', type=int, default=300, help='Job timeout in seconds')
    parser.add_argument('--gpu-id', type=int, default=None, help='GPU ID to use (sets CUDA_VISIBLE_DEVICES)')
    
    args = parser.parse_args()
    
    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        logger.info(f"Set CUDA_VISIBLE_DEVICES={args.gpu_id}")
    
    config = Config(
        csv_path=args.csv,
        workflow_path=args.workflow,
        output_dir=args.output_dir,
        comfyui_url=args.comfyui_url,
        fix_column=args.fix_column,
        fix_value=args.fix_value,
        ref_angle_col=args.ref_angle_col,
        front_angle_col=args.front_angle_col,
        original_img_col=args.original_img_col,
        swapped_img_col=args.swapped_img_col,
        generated_img_col=args.generated_img_col,
        batch_size=args.batch_size,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
        gpu_id=args.gpu_id
    )
    
    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == '__main__':
    main()