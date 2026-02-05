import json
import csv
import os
import sys
import argparse
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse
import uuid

def load_workflow(workflow_path: str) -> Dict:
    with open(workflow_path, 'r') as f:
        return json.load(f)

def load_csv(csv_path: str) -> List[Dict]:
    with open(csv_path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def download_image(url: str, output_dir: Path) -> Optional[str]:
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        ext = Path(urlparse(url).path).suffix or '.jpg'
        filename = f"{uuid.uuid4()}{ext}"
        filepath = output_dir / filename
        with open(filepath, 'wb') as f:
            f.write(response.content)
        return str(filepath)
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None

def update_workflow(workflow: Dict, original_img: str, generated_img: str, 
                   ref_angle: Optional[str], front_angle: Optional[str], 
                   prompt: str) -> Dict:
    wf = workflow.copy()
    
    for node_id, node in wf.items():
        if node.get("class_type") == "LoadImage":
            if "inputs" in node and "image" in node["inputs"]:
                node["inputs"]["image"] = original_img
        
        if node.get("class_type") == "LoadImage" and "reference" in str(node).lower():
            if "inputs" in node:
                node["inputs"]["image"] = generated_img
        
        if node.get("class_type") in ["CLIPTextEncode", "PromptNode"]:
            if "inputs" in node and "text" in node["inputs"]:
                node["inputs"]["text"] = prompt
    
    return wf

def submit_workflow(api_url: str, workflow: Dict) -> Optional[str]:
    try:
        response = requests.post(f"{api_url}/prompt", json={"prompt": workflow}, timeout=10)
        response.raise_for_status()
        return response.json().get("prompt_id")
    except Exception as e:
        print(f"Failed to submit workflow: {e}")
        return None

def poll_result(api_url: str, prompt_id: str, timeout: int = 600) -> Optional[Dict]:
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(f"{api_url}/history/{prompt_id}", timeout=10)
            response.raise_for_status()
            data = response.json()
            if prompt_id in data and data[prompt_id].get("status", {}).get("completed", False):
                return data[prompt_id]
        except Exception as e:
            print(f"Polling error: {e}")
        time.sleep(5)
    return None

def extract_output_images(history: Dict, api_url: str, output_dir: Path) -> List[str]:
    outputs = []
    for node_id, node_output in history.get("outputs", {}).items():
        if "images" in node_output:
            for img in node_output["images"]:
                filename = img.get("filename")
                subfolder = img.get("subfolder", "")
                img_url = f"{api_url}/view?filename={filename}&subfolder={subfolder}"
                local_path = download_image(img_url, output_dir)
                if local_path:
                    outputs.append(local_path)
    return outputs

def process_row(row: Dict, workflow: Dict, api_url: str, 
                download_dir: Path, output_dir: Path) -> Optional[str]:
    original_url = row.get("Original Image")
    generated_url = row.get("Generated Image")
    ref_angle_url = row.get("Reference Angle")
    front_angle_url = row.get("Front Angle")
    prompt = row.get("Edit Prompt", "")
    
    if not original_url or not generated_url:
        print("Missing required image URLs")
        return None
    
    original_local = download_image(original_url, download_dir)
    generated_local = download_image(generated_url, download_dir)
    
    if not original_local or not generated_local:
        return None
    
    ref_local = download_image(ref_angle_url, download_dir) if ref_angle_url else None
    front_local = download_image(front_angle_url, download_dir) if front_angle_url else None
    
    updated_wf = update_workflow(workflow, original_local, generated_local, 
                                 ref_local, front_local, prompt)
    
    prompt_id = submit_workflow(api_url, updated_wf)
    if not prompt_id:
        return None
    
    print(f"Submitted job {prompt_id}")
    history = poll_result(api_url, prompt_id)
    
    if not history:
        print(f"Job {prompt_id} timed out")
        return None
    
    result_images = extract_output_images(history, api_url, output_dir)
    return result_images[0] if result_images else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--workflow", required=True, help="Path to ComfyUI workflow JSON")
    parser.add_argument("--api-url", default="http://127.0.0.1:8188", help="ComfyUI API URL")
    parser.add_argument("--output-dir", default="./outputs", help="Output directory")
    parser.add_argument("--download-dir", default="./downloads", help="Download directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    download_dir = Path(args.download_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)
    
    workflow = load_workflow(args.workflow)
    rows = load_csv(args.csv)
    
    results = []
    for idx, row in enumerate(rows):
        print(f"Processing row {idx+1}/{len(rows)}")
        result_path = process_row(row, workflow, args.api_url, download_dir, output_dir)
        results.append({**row, "Result Path": result_path or "FAILED"})
    
    output_csv = output_dir / "results.csv"
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    main()