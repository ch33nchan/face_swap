# gradio_char_swap.py

import os
import tempfile
from typing import List, Any

import gradio as gr
import fal_client
from PIL import Image
import requests


FAL_MODEL = "fal-ai/nano-banana-pro/edit"

EDIT_PROMPT = """
ONLY the character's visual appearance in Image 1. DO NOT change pose, position, or composition.

CRITICAL - NEVER CHANGE (from Image 1):
- Character's EXACT body pose, arm positions, leg positions, head tilt, body angle
- Character's EXACT position in the frame (where they are standing/sitting)
- Background, scene, environment - must be PIXEL PERFECT same
- Camera angle, framing, composition, crop
- Lighting and shadows
- Composition

SWAP ONLY these visual features FROM Image 2:
- Face shape and facial features
- Hair color, style, and length
- Clothing appearance (colors, patterns, open/closed state of clothes eg: jacket open vs closed, style)
- Visible accessories (jewelry, glasses, hats, etc.)
- Facial expression fix: Adjust eye gaze to look directly forward at the camera, close the mouth to a slightly pursed closed state, straighten the head to face directly forward without tilting or turning left, and set the expression to a concerned serious neutral look.
- Exact POSE of the character in image 1 must be preserved.
Think of it like a costume change on the EXACT same character pose.
The silhouette and position should be IDENTICAL to Image 1.

Output: Image 1 with ONLY the character's visual appearance changed to match Image 2.REALISTIC STYLE
""".strip()


def _normalize_gallery_item(item: Any) -> Image.Image:
    if isinstance(item, Image.Image):
        return item
    if isinstance(item, tuple):
        return Image.open(item[0])
    if isinstance(item, str):
        return Image.open(item)
    raise TypeError(f"Unsupported gallery item type: {type(item)}")


def _save_temp_image(img: Image.Image) -> str:
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    img.save(path)
    return path


def _upload_image(img: Image.Image) -> str:
    temp_path = _save_temp_image(img)
    url = fal_client.upload_file(temp_path)
    os.unlink(temp_path)
    return url


def run_edit(
    base_image: Image.Image,
    ref_main: Image.Image,
    ref_angles: List[Any],
    strength: float,
) -> Image.Image:
    if base_image is None:
        raise ValueError("Base image (Image 1) is required")
    if ref_main is None:
        raise ValueError("Reference image (Image 2) is required")

    base_url = _upload_image(base_image)
    ref_urls = [_upload_image(ref_main)]

    if ref_angles:
        for item in ref_angles:
            img = _normalize_gallery_item(item)
            ref_urls.append(_upload_image(img))

    inputs = {
        "prompt": EDIT_PROMPT,
        "image_url": base_url,
        "image_urls": ref_urls,
        "strength": strength,
    }

    handle = fal_client.submit(FAL_MODEL, arguments=inputs)
    result = handle.get()

    output_url = result["images"][0]["url"]
    resp = requests.get(output_url, timeout=60)
    resp.raise_for_status()

    out_fd, out_path = tempfile.mkstemp(suffix=".png")
    os.close(out_fd)
    with open(out_path, "wb") as f:
        f.write(resp.content)

    return Image.open(out_path)


with gr.Blocks(title="Character Appearance Swap") as demo:
    gr.Markdown("## Character Appearance Swap (Pose & Background from Image 1, Appearance from Image 2)")
    
    with gr.Row():
        with gr.Column():
            base_image = gr.Image(
                label="Image 1: Base (Keeps Pose, Position, Background)",
                type="pil",
            )
        with gr.Column():
            ref_main = gr.Image(
                label="Image 2: New Character Appearance (Front/Main Reference)",
                type="pil",
            )

    ref_angles = gr.Gallery(
        label="Additional Reference Angles for Image 2 (Optional)",
        columns=4,
        height=200,
    )

    strength = gr.Slider(
        minimum=0.5,
        maximum=1.0,
        value=1.0,
        step=0.05,
        label="Edit Strength",
    )

    run_btn = gr.Button("Run Character Swap", variant="primary")
    output = gr.Image(label="Output", type="pil")

    run_btn.click(
        fn=run_edit,
        inputs=[base_image, ref_main, ref_angles, strength],
        outputs=output,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)