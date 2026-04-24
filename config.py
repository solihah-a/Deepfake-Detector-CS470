from pathlib import Path

DATASET_NAME = "Rajarshi-Roy-research/Defactify_Image_Dataset"
IMAGE_COLUMN = "Image"
LABEL_COLUMN = "Label_A"   # binary label: real vs AI-generated

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 0.001
NUM_CLASSES = 2
SEED = 42

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "best_model.pt"