import os
import sys
import torch
import numpy as np
from PIL import Image
from .config import SAM2_DIR, CHECKPOINT_PATH, MODEL_CFG, SAM2_POINTS_PER_SIDE, SAM2_PRED_IOU_THRESH, SAM2_STABILITY_SCORE_THRESH, SAM2_CROP_N_LAYERS, SAM2_CROP_N_POINTS_DOWNSCALE, SAM2_MIN_MASK_REGION_AREA

# Add SAM2 to Python path
sys.path.insert(0, SAM2_DIR)

# Change to SAM2 directory for proper config loading
original_cwd = os.getcwd()
os.chdir(SAM2_DIR)

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    
    # SAM2 model configuration from config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Build SAM2 model using absolute checkpoint path
    sam2_model = build_sam2(MODEL_CFG, CHECKPOINT_PATH, device=device)
    
    # Use automatic mask generator for multiple objects
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=SAM2_POINTS_PER_SIDE,
        pred_iou_thresh=SAM2_PRED_IOU_THRESH,
        stability_score_thresh=SAM2_STABILITY_SCORE_THRESH,
        crop_n_layers=SAM2_CROP_N_LAYERS,
        crop_n_points_downscale_factor=SAM2_CROP_N_POINTS_DOWNSCALE,
        min_mask_region_area=SAM2_MIN_MASK_REGION_AREA,  # Filter out small regions
    )
    
    print(f"✅ SAM2 loaded successfully on {device}")
    
finally:
    # Return to original directory
    os.chdir(original_cwd)

def segment_image(image_path):
    """
    Segment furniture in image using SAM2 automatic mask generation.
    Returns paths to all generated mask PNG files for different furniture parts.
    """
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        
        # Generate masks for all objects in the image
        masks_data = mask_generator.generate(image_array)
        
        if not masks_data:
            return None
        
        # Sort masks by area (largest first) and take top segments
        masks_data = sorted(masks_data, key=lambda x: x['area'], reverse=True)
        
        # Take top 5-8 largest segments (doors, handles, panels, etc.)
        # top_masks = masks_data[:6]  # Adjust this number based on your needs
        top_masks = masks_data  # Adjust this number based on your needs
        
        mask_paths = []
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        for i, mask_data in enumerate(top_masks):
            # Get the segmentation mask
            mask = mask_data['segmentation']
            
            # Convert to PIL Image
            mask_image = Image.fromarray((mask * 255).astype(np.uint8))
            
            # Save each mask with a unique name
            mask_filename = f"{base_name}_mask_{i+1}.png"
            mask_path = os.path.join(os.path.dirname(image_path), mask_filename)
            mask_image.save(mask_path)
            
            mask_paths.append(mask_path)
            
            # Print info about each segment
            area = mask_data['area']
            stability_score = mask_data['stability_score']
            print(f"✅ Segment {i+1}: area={area}, stability={stability_score:.3f}")
        
        print(f"✅ Generated {len(mask_paths)} furniture part masks!")
        return mask_paths
        
    except Exception as e:
        print(f"❌ Segmentation error: {e}")
        return None