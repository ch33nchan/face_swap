import os
import io
import base64
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image
import logging

from src.face_swap import FaceSwapPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Face Swap API", version="1.0.0")

pipeline: Optional[FaceSwapPipeline] = None


class SwapRequest(BaseModel):
    prompt: str = Field(default="high quality portrait, detailed face, natural lighting")
    negative_prompt: str = Field(default="blurry, low quality, distorted face, artifacts")
    num_inference_steps: int = Field(default=28, ge=1, le=100)
    guidance_scale: float = Field(default=3.5, ge=0.0, le=20.0)
    strength: float = Field(default=0.75, ge=0.0, le=1.0)
    seed: Optional[int] = Field(default=None)


class InitRequest(BaseModel):
    model_id: str = Field(default="black-forest-labs/FLUX.1-dev")
    lora_path: Optional[str] = Field(default=None)
    lora_scale: float = Field(default=1.0, ge=0.0, le=2.0)


@app.on_event("startup")
async def startup_event():
    global pipeline
    model_id = os.getenv("MODEL_ID", "black-forest-labs/FLUX.1-dev")
    lora_path = os.getenv("LORA_PATH", None)
    lora_scale = float(os.getenv("LORA_SCALE", "1.0"))
    
    logger.info("Initializing Face Swap Pipeline")
    pipeline = FaceSwapPipeline(
        model_id=model_id,
        lora_path=lora_path,
        lora_scale=lora_scale,
    )
    logger.info("Pipeline ready")


@app.post("/init")
async def initialize_pipeline(request: InitRequest):
    global pipeline
    try:
        if pipeline is not None:
            pipeline.unload()
        
        pipeline = FaceSwapPipeline(
            model_id=request.model_id,
            lora_path=request.lora_path,
            lora_scale=request.lora_scale,
        )
        return {"status": "success", "message": "Pipeline initialized"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/swap")
async def swap_face(
    base_image: UploadFile = File(...),
    reference_image: UploadFile = File(...),
    prompt: str = Form("high quality portrait, detailed face, natural lighting"),
    negative_prompt: str = Form("blurry, low quality, distorted face, artifacts"),
    num_inference_steps: int = Form(28),
    guidance_scale: float = Form(3.5),
    strength: float = Form(0.75),
    seed: Optional[int] = Form(None),
):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        base_img = Image.open(io.BytesIO(await base_image.read())).convert("RGB")
        ref_img = Image.open(io.BytesIO(await reference_image.read())).convert("RGB")
        
        result = pipeline.swap_face(
            base_image=base_img,
            reference_image=ref_img,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            seed=seed,
        )
        
        buffered = io.BytesIO()
        result.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return JSONResponse({
            "status": "success",
            "image": img_str,
            "format": "png"
        })
    
    except Exception as e:
        logger.error(f"Face swap failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "pipeline_loaded": pipeline is not None
    }


@app.post("/unload")
async def unload_pipeline():
    global pipeline
    if pipeline is not None:
        pipeline.unload()
        pipeline = None
        return {"status": "success", "message": "Pipeline unloaded"}
    return {"status": "success", "message": "No pipeline to unload"}
