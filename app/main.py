from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import shutil
import os
from dotenv import load_dotenv
from app.sam2_handler import segment_color_based, segment_sam2_auto, segment_sam2_interactive
from app.config import TEMP_DIR

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Furniture Color Preview Backend", 
    description="API for segmenting furniture in images using multiple strategies",
    version="2.0.0"
)

class InteractivePrompts(BaseModel):
    points: list[list[int]] | None = None
    boxes: list[list[int]] | None = None
    labels: list[int] | None = None

@app.post("/segment/color")
async def segment_color(file: UploadFile = File(...)):
    """
    Segment furniture using color-based clustering in LAB color space.
    Best for: Clear furniture images with distinct color regions.
    Fast and reliable for most furniture segmentation tasks.
    """
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PNG, JPG, JPEG allowed.")
    
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    temp_image_path = os.path.join(TEMP_DIR, file.filename)
    with open(temp_image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    mask_paths = segment_color_based(temp_image_path)
    
    if mask_paths:
        return JSONResponse({
            "success": True,
            "strategy": "color-based",
            "description": "K-means clustering in LAB color space",
            "message": f"Generated {len(mask_paths)} color region masks",
            "masks": [
                {
                    "id": i+1,
                    "filename": os.path.basename(path),
                    "path": path,
                    "download_url": f"/download/{os.path.basename(path)}"
                } for i, path in enumerate(mask_paths)
            ]
        })
    else:
        raise HTTPException(status_code=500, detail="Color-based segmentation failed.")

@app.post("/segment/sam2-auto")
async def segment_sam2_automatic(file: UploadFile = File(...)):
    """
    Segment furniture using SAM2 automatic mask generation.
    Best for: Complex furniture with irregular shapes, detailed segmentation needed.
    Slower but more detailed than color-based method.
    """
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PNG, JPG, JPEG allowed.")
    
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    temp_image_path = os.path.join(TEMP_DIR, file.filename)
    with open(temp_image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    mask_paths = segment_sam2_auto(temp_image_path)
    
    if mask_paths:
        return JSONResponse({
            "success": True,
            "strategy": "sam2-automatic",
            "description": "SAM2 automatic mask generation with stability scoring",
            "message": f"Generated {len(mask_paths)} SAM2 masks",
            "masks": [
                {
                    "id": i+1,
                    "filename": os.path.basename(path),
                    "path": path,
                    "download_url": f"/download/{os.path.basename(path)}"
                } for i, path in enumerate(mask_paths)
            ]
        })
    else:
        raise HTTPException(status_code=500, detail="SAM2 automatic segmentation failed.")

@app.post("/segment/sam2-interactive")
async def segment_sam2_interactive_endpoint(
    file: UploadFile = File(...),
    prompts: InteractivePrompts = Body(...)
):
    """
    Segment furniture using SAM2 interactive mode with user prompts.
    Best for: Precise control over specific furniture parts.
    
    Body parameters:
    - points: [[x,y], [x,y]] - Click coordinates for foreground/background
    - boxes: [[x1,y1,x2,y2]] - Drag boxes around regions of interest  
    - labels: [1, 0, 1] - 1=foreground, 0=background (same length as points)
    """
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PNG, JPG, JPEG allowed.")
    
    # Validate prompts
    if not prompts.points and not prompts.boxes:
        raise HTTPException(status_code=400, detail="Either points or boxes must be provided")
    
    if prompts.points and prompts.labels:
        if len(prompts.points) != len(prompts.labels):
            raise HTTPException(status_code=400, detail="Number of points must match number of labels")
    
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    temp_image_path = os.path.join(TEMP_DIR, file.filename)
    with open(temp_image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    mask_paths = segment_sam2_interactive(
        temp_image_path,
        points=prompts.points,
        boxes=prompts.boxes,
        labels=prompts.labels
    )
    
    if mask_paths:
        return JSONResponse({
            "success": True,
            "strategy": "sam2-interactive",
            "description": "SAM2 with user-guided prompts",
            "prompts_used": {
                "points": len(prompts.points) if prompts.points else 0,
                "boxes": len(prompts.boxes) if prompts.boxes else 0,
                "labels": len(prompts.labels) if prompts.labels else 0
            },
            "message": f"Generated {len(mask_paths)} interactive masks",
            "masks": [
                {
                    "id": i+1,
                    "filename": os.path.basename(path),
                    "path": path,
                    "download_url": f"/download/{os.path.basename(path)}"
                } for i, path in enumerate(mask_paths)
            ]
        })
    else:
        raise HTTPException(status_code=500, detail="SAM2 interactive segmentation failed.")

@app.get("/download/{filename}")
async def download_mask(filename: str):
    """
    Download a specific mask file by filename.
    """
    file_path = os.path.join(TEMP_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Mask file not found.")
    
    return FileResponse(file_path, media_type='image/png', filename=filename)