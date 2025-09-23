import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# SAM2 Configuration - Use environment variables for security
SAM2_DIR = os.getenv("SAM2_DIR", "/path/to/sam2")
MODEL_SIZE = os.getenv("MODEL_SIZE", "large")

# Construct model paths
CHECKPOINT_PATH = os.path.join(SAM2_DIR, "checkpoints", f"sam2.1_hiera_{MODEL_SIZE}.pt")
MODEL_CFG = f"configs/sam2.1/sam2.1_hiera_{MODEL_SIZE[0]}.yaml"

# Handle base_plus model special case
if MODEL_SIZE == "base_plus":
    CHECKPOINT_PATH = os.path.join(SAM2_DIR, "checkpoints", "sam2.1_hiera_b+.pt")
    MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

# Application settings
TEMP_DIR = os.path.join(BASE_DIR, "temp")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
TEMP_RETENTION_HOURS = int(os.getenv("TEMP_RETENTION_HOURS", "24"))

# Ensure temp directory exists
Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)