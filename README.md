# Furniture Color Preview Backend

A FastAPI backend service that uses SAM2 (Segment Anything Model 2) to segment furniture images into multiple parts, enabling color customization for furniture visualization.

## Features

- ⚙️ **Multiple Segmentation Strategies:**
  - **Color-Based**: Fast clustering for simple furniture (primary method)
  - **SAM2 Automatic**: Detailed AI-powered segmentation (fallback)
  - **SAM2 Interactive**: Precise user-guided segmentation (precision control)
- 🎯 **Smart Fallback System**: Automatically tries better methods if primary fails
- ⚡ **CUDA Acceleration**: GPU-powered processing for fast results
- 🚀 **REST API**: Clean HTTP endpoints with detailed responses
- 📱 **Mobile App Ready**: Designed for furniture customization apps
- 🔧 **Configurable**: Environment-based settings for all parameters

## API Endpoints

### POST `/segment/color`
**Color-Based Segmentation** - Fast and reliable for most furniture images.

- **Best for:** Clear furniture with distinct color regions
- **Method:** K-means clustering in LAB color space
- **Speed:** Fast (seconds)
- **Accuracy:** Good for simple furniture

**Request:**
```bash
curl -X POST "http://localhost:8000/segment/color" \
  -F "file=@sofa.jpg"
```

**Response:**
```json
{
    "success": true,
    "strategy": "color-based",
    "description": "K-means clustering in LAB color space",
    "message": "Generated 3 color region masks",
    "masks": [
        {
            "id": 1,
            "filename": "sofa_color_1_mask.png",
            "download_url": "/download/sofa_color_1_mask.png"
        }
    ]
}
```

### POST `/segment/sam2-auto`
**SAM2 Automatic Segmentation** - Detailed segmentation for complex furniture.

- **Best for:** Irregular shapes, detailed furniture parts
- **Method:** SAM2 automatic mask generation
- **Speed:** Slower (10-30 seconds)
- **Accuracy:** High, with stability scoring

**Request:**
```bash
curl -X POST "http://localhost:8000/segment/sam2-auto" \
  -F "file=@wardrobe.jpg"
```

### POST `/segment/sam2-interactive`
**SAM2 Interactive Segmentation** - Precise control with user prompts.

- **Best for:** When you need exact control over specific parts
- **Method:** User-guided point and box prompts
- **Speed:** Fast (5-15 seconds)
- **Accuracy:** Highest precision

**Request:**
```json
{
  "points": [[500, 800], [200, 400]],
  "labels": [1, 1]
}
```

```bash
curl -X POST "http://localhost:8000/segment/sam2-interactive" \
  -F "file=@sofa.jpg" \
  -H "Content-Type: application/json" \
  -d '{"points": [[500, 800]], "labels": [1]}'
```

**Prompt Types:**
- `points`: `[[x,y], [x,y]]` - Click coordinates
- `boxes`: `[[x1,y1,x2,y2]]` - Drag rectangles
- `labels`: `[1, 0, 1]` - 1=foreground, 0=background

### GET `/download/{filename}`
Download a specific mask file.

```bash
curl -o mask.png "http://localhost:8000/download/sofa_color_1_mask.png"
```

## Setup

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)
- SAM2 model installed

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd furniture-design-backend
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install SAM2**
```bash
# Clone and install SAM2 separately
git clone https://github.com/facebookresearch/segment-anything-2.git sam2
cd sam2
pip install -e .
cd checkpoints && ./download_ckpts.sh
```

5. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your SAM2 path and settings
```

6. **Run the server**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SAM2_DIR` | `/path/to/sam2` | Path to SAM2 installation |
| `MODEL_SIZE` | `large` | SAM2 model size (tiny/small/base_plus/large) |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `TEMP_RETENTION_HOURS` | `24` | How long to keep temporary files |

## Usage Example

```bash
# Upload an image and get segmentation masks
curl -X POST "http://localhost:8000/segment/color" \
  -F "file=@furniture_image.jpg" | jq

# Download a specific mask
curl -o mask1.png "http://localhost:8000/download/furniture_mask_1.png"
```

## Project Structure

```
furniture-design-backend/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   ├── config.py         # Configuration settings
│   └── sam2_handler.py   # SAM2 model integration
├── temp/                 # Temporary files (gitignored)
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
└── README.md
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
