import os
import sys
import torch
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from scipy import ndimage
from skimage import color
from .config import SAM2_DIR, CHECKPOINT_PATH, MODEL_CFG, SAM2_POINTS_PER_SIDE, SAM2_PRED_IOU_THRESH, SAM2_STABILITY_SCORE_THRESH, SAM2_CROP_N_LAYERS, SAM2_CROP_N_POINTS_DOWNSCALE, SAM2_MIN_MASK_REGION_AREA, TEMP_DIR

# Constants for image processing
BINARY_THRESHOLD = 127
WHITE_THRESHOLD = 240

# Add SAM2 to Python path
sys.path.insert(0, SAM2_DIR)

original_cwd = os.getcwd()
os.chdir(SAM2_DIR)

try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2_model = build_sam2(MODEL_CFG, CHECKPOINT_PATH, device=device)

    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=SAM2_POINTS_PER_SIDE,
        pred_iou_thresh=SAM2_PRED_IOU_THRESH,
        stability_score_thresh=SAM2_STABILITY_SCORE_THRESH,
        crop_n_layers=SAM2_CROP_N_LAYERS,
        crop_n_points_downscale_factor=SAM2_CROP_N_POINTS_DOWNSCALE,
        min_mask_region_area=SAM2_MIN_MASK_REGION_AREA,
    )

    print(f"✅ SAM2 loaded on {device}")

finally:
    os.chdir(original_cwd)

def segment_color_based(image_path):
    """
    Segment furniture using color-based clustering in LAB space.
    Returns paths to PNG mask files for different color regions.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        height, width = image_array.shape[:2]

        # Convert to LAB color space for better color clustering
        lab_image = color.rgb2lab(image_array)
        pixels = lab_image.reshape(-1, 3)

        # K-means clustering for dominant colors
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        kmeans.fit(pixels)

        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        label_image = labels.reshape(height, width)

        mask_paths = []
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        for i, center in enumerate(centers):
            rgb_center = color.lab2rgb(center.reshape(1, 1, 3)).reshape(3) * 255
            r, g, b = rgb_center

            # Skip white background
            if r > WHITE_THRESHOLD and g > WHITE_THRESHOLD and b > WHITE_THRESHOLD:
                continue

            color_name = f"color_{i+1}"
            mask = (label_image == i).astype(np.uint8) * 255

            # Minimal morphological cleaning
            mask = ndimage.binary_opening(mask > BINARY_THRESHOLD, structure=np.ones((2,2))).astype(np.uint8) * 255
            mask = ndimage.binary_closing(mask > BINARY_THRESHOLD, structure=np.ones((2,2))).astype(np.uint8) * 255

            mask_image = Image.fromarray(mask)
            mask_filename = f"{base_name}_{color_name}_mask.png"
            mask_path = os.path.join(TEMP_DIR, mask_filename)
            mask_image.save(mask_path)

            mask_paths.append(mask_path)
            print(f"✅ Created {color_name} mask: RGB({r:.0f}, {g:.0f}, {b:.0f})")

        print(f"✅ Color-based segmentation: {len(mask_paths)} masks generated")
        return mask_paths

    except Exception as e:
        print(f"❌ Color segmentation error: {e}")
        return None

def segment_sam2_interactive(image_path, points=None, boxes=None, labels=None):
    """
    Segment using SAM2 interactive mode with point/box prompts.
    Returns paths to PNG mask files based on user prompts.

    Args:
        image_path: Path to input image
        points: List of [x,y] coordinates for point prompts
        boxes: List of [x1,y1,x2,y2] coordinates for box prompts
        labels: List of labels (1=foreground, 0=background) for points
    """
    try:
        # Import SAM2 predictor
        from sam2.sam2_image_predictor import Sam2ImagePredictor

        # Load SAM2 model for interactive mode
        predictor_model = build_sam2(MODEL_CFG, CHECKPOINT_PATH, device=device)
        predictor = Sam2ImagePredictor(predictor_model)

        # Load image
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)

        # Set image in predictor
        predictor.set_image(image_array)

        # Prepare prompts
        point_coords = np.array(points) if points else None
        box_coords = np.array(boxes) if boxes else None
        point_labels = np.array(labels) if labels else None

        # Generate masks
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_coords,
            multimask_output=True
        )

        # Convert to mask images
        mask_paths = []
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        for i in range(masks.shape[0]):
            mask = masks[i].astype(np.uint8) * 255
            mask_image = Image.fromarray(mask)

            mask_filename = f"{base_name}_interactive_mask_{i+1}.png"
            mask_path = os.path.join(TEMP_DIR, mask_filename)
            mask_image.save(mask_path)

            mask_paths.append(mask_path)
            confidence = scores[i]
            print(f"✅ Interactive mask {i+1}: confidence={confidence:.3f}")

        print(f"✅ SAM2 interactive segmentation: {len(mask_paths)} masks generated")
        return mask_paths

    except Exception as e:
        print(f"❌ SAM2 interactive error: {e}")
        return None

def segment_sam2_auto(image_path):
    """
    Segment using SAM2 automatic mask generation.
    Returns paths to PNG mask files for detected regions.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)

        masks_data = mask_generator.generate(image_array)
        if not masks_data:
            return None

        masks_data = sorted(masks_data, key=lambda x: x['area'], reverse=True)
        # top_masks = masks_data[:10]  # Limit to top 10 masks
        top_masks = masks_data

        mask_paths = []
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        for i, mask_data in enumerate(top_masks):
            mask = mask_data['segmentation']
            mask_image = Image.fromarray((mask * 255).astype(np.uint8))

            mask_filename = f"{base_name}_sam2_mask_{i+1}.png"
            mask_path = os.path.join(TEMP_DIR, mask_filename)
            mask_image.save(mask_path)

            mask_paths.append(mask_path)
            area = mask_data['area']
            stability = mask_data['stability_score']
            print(f"✅ SAM2 Segment {i+1}: area={area}, stability={stability:.3f}")

        print(f"✅ SAM2 automatic segmentation: {len(mask_paths)} masks generated")
        return mask_paths

    except Exception as e:
        print(f"❌ SAM2 auto error: {e}")
        return None