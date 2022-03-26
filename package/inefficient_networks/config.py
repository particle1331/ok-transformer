from pydantic import BaseModel
from pathlib import Path


class Config(BaseModel):
    DATASET_DIR = Path(__file__).parent.resolve() / "data"
    TRAINED_MODELS_DIR = Path(__file__).parent.resolve() / "trained_models"
    
    # Create directories
    TRAINED_MODELS_DIR.mkdir(exist_ok=True)
    DATASET_DIR.mkdir(exist_ok=True)


config = Config()