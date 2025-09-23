from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import shutil
import os
from dotenv import load_dotenv
from app.sam2_handler import segment_image
from app.config import TEMP_DIR

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Furniture Color Preview Backend", 
    description="API for segmenting furniture in images using SAM2",
    version="1.0.0"
)

@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    """
    Accepts an image file via multipart upload, segments the furniture using SAM2,
    and returns paths to multiple mask images for different furniture parts.
    """
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PNG, JPG, JPEG allowed.")
    
    # Ensure temp directory exists
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Save uploaded image temporarily
    temp_image_path = os.path.join(TEMP_DIR, file.filename)
    with open(temp_image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Perform segmentation
    mask_paths = segment_image(temp_image_path)
    
    if mask_paths:
        # Return JSON with all mask file paths
        return JSONResponse({
            "success": True,
            "message": f"Generated {len(mask_paths)} furniture part masks",
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
        raise HTTPException(status_code=500, detail="Segmentation failed. No masks generated.")

@app.get("/download/{filename}")
async def download_mask(filename: str):
    """
    Download a specific mask file by filename.
    """
    file_path = os.path.join(TEMP_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Mask file not found.")
    
    return FileResponse(file_path, media_type='image/png', filename=filename)