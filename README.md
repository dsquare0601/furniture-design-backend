# Furniture Color Preview Backend

A FastAPI backend service that uses SAM2 (Segment Anything Model 2) to segment furniture images into multiple parts, enabling color customization for furniture visualization.

## Features

- 🎯 **Multi-part Segmentation**: Automatically detects and segments different furniture components (doors, handles, panels, etc.)
- 🎨 **Color Customization Ready**: Generates separate masks for each furniture part
- ⚡ **Fast Processing**: CUDA-accelerated SAM2 model
- 🚀 **REST API**: Simple HTTP endpoints for easy integration
- 📱 **Mobile Ready**: Designed for Flutter mobile app integration

## API Endpoints

### POST `/segment`
Upload a furniture image and get multiple mask files for different parts.

**Request:**
- Multipart form data with `file` field
- Supported formats: PNG, JPG, JPEG

**Response:**
```json
{
    "success": true,
    "message": "Generated 6 furniture part masks",
    "masks": [
        {
            "id": 1,
            "filename": "furniture_mask_1.png",
            "path": "/path/to/mask",
            "download_url": "/download/furniture_mask_1.png"
        }
    ]
}
```

### GET `/download/{filename}`
Download a specific mask file.

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
curl -X POST "http://localhost:8000/segment" \
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
