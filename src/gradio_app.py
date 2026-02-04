import os
import gradio as gr
from PIL import Image
import torch
import logging

from src.face_swap import FaceSwapPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pipeline = None


def initialize_pipeline(model_id: str, lora_path: str = None, lora_scale: float = 1.0):
    global pipeline
    try:
        if pipeline is not None:
            pipeline.unload()
        
        pipeline = FaceSwapPipeline(
            model_id=model_id,
            lora_path=lora_path if lora_path else None,
            lora_scale=lora_scale,
        )
        return "Pipeline initialized successfully"
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return f"Error: {str(e)}"


def perform_swap(
    base_image: Image.Image,
    reference_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    strength: float,
    seed: int,
):
    if pipeline is None:
        return None, "Pipeline not initialized. Please initialize first."
    
    try:
        result = pipeline.swap_face(
            base_image=base_image,
            reference_image=reference_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            strength=strength,
            seed=seed if seed > 0 else None,
        )
        return result, "Success"
    except Exception as e:
        logger.error(f"Face swap failed: {e}")
        return None, f"Error: {str(e)}"


def create_ui():
    with gr.Blocks(title="Face Swap with FLUX") as demo:
        gr.Markdown("# Face Swap Character Transfer")
        
        with gr.Tab("Initialize"):
            with gr.Row():
                model_id = gr.Textbox(
                    value="black-forest-labs/FLUX.1-dev",
                    label="Model ID",
                )
                lora_path = gr.Textbox(
                    value="",
                    label="LORA Path (optional)",
                )
                lora_scale = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=1.0,
                    label="LORA Scale",
                )
            
            init_btn = gr.Button("Initialize Pipeline", variant="primary")
            init_status = gr.Textbox(label="Status", interactive=False)
            
            init_btn.click(
                fn=initialize_pipeline,
                inputs=[model_id, lora_path, lora_scale],
                outputs=init_status,
            )
        
        with gr.Tab("Face Swap"):
            with gr.Row():
                with gr.Column():
                    base_image = gr.Image(type="pil", label="Base Image")
                    reference_image = gr.Image(type="pil", label="Reference Face")
                
                with gr.Column():
                    output_image = gr.Image(type="pil", label="Result")
                    status_text = gr.Textbox(label="Status", interactive=False)
            
            with gr.Row():
                prompt = gr.Textbox(
                    value="high quality portrait, detailed face, natural lighting",
                    label="Prompt",
                )
                negative_prompt = gr.Textbox(
                    value="blurry, low quality, distorted face, artifacts",
                    label="Negative Prompt",
                )
            
            with gr.Row():
                steps = gr.Slider(1, 100, value=28, step=1, label="Steps")
                guidance_scale = gr.Slider(0.0, 20.0, value=3.5, step=0.1, label="Guidance Scale")
                strength = gr.Slider(0.0, 1.0, value=0.75, step=0.05, label="Strength")
                seed = gr.Number(value=-1, label="Seed (-1 for random)")
            
            swap_btn = gr.Button("Swap Face", variant="primary")
            
            swap_btn.click(
                fn=perform_swap,
                inputs=[
                    base_image,
                    reference_image,
                    prompt,
                    negative_prompt,
                    steps,
                    guidance_scale,
                    strength,
                    seed,
                ],
                outputs=[output_image, status_text],
            )
        
        gr.Markdown("""
        ## Instructions
        1. **Initialize**: Set model and LORA path, then click Initialize
        2. **Face Swap**: Upload base image and reference face, adjust parameters, click Swap Face
        
        ### Dataset for Training
        - Collect 20-50 high-quality images of the reference character
        - Images should have clear, frontal or varied angles of the face
        - Different expressions, lighting conditions
        - Resolution: at least 512x512
        """)
    
    return demo


def main():
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
