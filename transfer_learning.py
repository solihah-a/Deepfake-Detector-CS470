"""
Author: Jason

This script runs a transfer learning experiment for deepfake detection.
It uses a pretrained CLIP-based model and fine-tunes only the attention head
on a prepared binary image dataset with real and fake classes.

Technical details:
- Images are loaded from train, validation, and test folders.
- Each split should contain real/ and fake/ subfolders.
- Images are resized to 224x224 and normalized with CLIP mean and std values.
- The pretrained backbone is frozen and only attention_head is trained.
- Training uses BCEWithLogitsLoss and the Adam optimizer.
- Validation loss is used to keep the best attention head.
- The script evaluates the model before and after transfer learning.
- It saves the before/after results to a CSV file and saves the best attention head.
"""

import csv
import random
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from project_config import dataset_presets, default_dataset_name, model_presets
from D3.models.clip_models import CLIPModelShuffleAttentionPenultimateLayer


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
THRESHOLD = 0.5
DATASET_NAME = default_dataset_name
MODEL_NAME = "d3_pretrained"
DATA_ROOT = Path(dataset_presets[DATASET_NAME]["prepared_data_dir"])
CHECKPOINT_PATH = Path(model_presets[MODEL_NAME]["checkpoint"])
OUTPUT_DIR = Path("outputs") / "transfer_learning" / "basic"
RESULTS_CSV_PATH = OUTPUT_DIR / "results.csv"
EPOCHS = 5
PATIENCE = 2
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
SEED = 418
DEVICE_NAME = "auto"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_images(folder):
    return sorted(
        path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
class RealFakeDataset(Dataset):
    def __init__(self, split_root, image_size=224):
        self.split_root = Path(split_root)
        self.real_dir = self.split_root / "real"
        self.fake_dir = self.split_root / "fake"
        self.samples = [(path, 0) for path in list_images(self.real_dir)]
        self.samples += [(path, 1) for path in list_images(self.fake_dir)]
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        return image, label


def create_loader(split_root, batch_size, num_workers, shuffle):
    dataset = RealFakeDataset(split_root)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def get_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def load_model(model_name, checkpoint_path, device):
    model_config = model_presets[model_name]
    model = CLIPModelShuffleAttentionPenultimateLayer(
        model_config["clip_name"],
        shuffle_times=model_config["shuffle_times"],
        original_times=model_config["original_times"],
        patch_size=[model_config["patch_size"]],
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.attention_head.load_state_dict(state_dict)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.attention_head.parameters():
        parameter.requires_grad = True
    model.to(device)
    return model


def calculate_metrics(y_true, y_pred):
    predicted_labels = (y_pred >= THRESHOLD).astype(np.int64)
    real_mask = y_true == 0
    fake_mask = y_true == 1
    real_acc = float(np.mean(predicted_labels[real_mask] == y_true[real_mask]))
    fake_acc = float(np.mean(predicted_labels[fake_mask] == y_true[fake_mask]))
    overall_acc = float(np.mean(predicted_labels == y_true))

    return {
        "num_real": int(real_mask.sum()),
        "num_fake": int(fake_mask.sum()),
        "real_accuracy_percent": real_acc * 100.0,
        "fake_accuracy_percent": fake_acc * 100.0,
        "overall_accuracy_percent": overall_acc * 100.0,
        "real_classified_as_real": int(np.sum(real_mask & (predicted_labels == 0))),
        "real_classified_as_fake": int(np.sum(real_mask & (predicted_labels == 1))),
        "fake_classified_as_real": int(np.sum(fake_mask & (predicted_labels == 0))),
        "fake_classified_as_fake": int(np.sum(fake_mask & (predicted_labels == 1))),
    }


def evaluate_model(model, loader, loss_fn, device, eval_seed):
    model.eval()
    torch.manual_seed(eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(eval_seed)
    y_true = []
    y_pred = []
    total_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).float()
            logits = model(images).flatten()
            loss = loss_fn(logits, labels)
            probs = torch.sigmoid(logits)
            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size
            y_true.extend(labels.cpu().numpy().astype(np.int64).tolist())
            y_pred.extend(probs.cpu().numpy().tolist())

    metrics = calculate_metrics(
        np.array(y_true, dtype=np.int64),
        np.array(y_pred, dtype=np.float32),
    )
    metrics["loss"] = total_loss / max(total_items, 1)
    return metrics


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.model.eval()
    model.attention_head.train()
    total_loss = 0.0
    total_items = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).float()
        optimizer.zero_grad()
        logits = model(images).flatten()
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
    return total_loss / max(total_items, 1)


def write_results_csv(results_path, before_metrics, after_metrics):
    rows = [
        {
            "stage": "before transfer learning",
            "overall_accuracy_percent": before_metrics["overall_accuracy_percent"],
            "real_accuracy_percent": before_metrics["real_accuracy_percent"],
            "fake_accuracy_percent": before_metrics["fake_accuracy_percent"],
            "real_images": before_metrics["num_real"],
            "fake_images": before_metrics["num_fake"],
            "real_classified_as_real": before_metrics["real_classified_as_real"],
            "real_classified_as_fake": before_metrics["real_classified_as_fake"],
            "fake_classified_as_real": before_metrics["fake_classified_as_real"],
            "fake_classified_as_fake": before_metrics["fake_classified_as_fake"],
        },
        {
            "stage": "after transfer learning",
            "overall_accuracy_percent": after_metrics["overall_accuracy_percent"],
            "real_accuracy_percent": after_metrics["real_accuracy_percent"],
            "fake_accuracy_percent": after_metrics["fake_accuracy_percent"],
            "real_images": after_metrics["num_real"],
            "fake_images": after_metrics["num_fake"],
            "real_classified_as_real": after_metrics["real_classified_as_real"],
            "real_classified_as_fake": after_metrics["real_classified_as_fake"],
            "fake_classified_as_real": after_metrics["fake_classified_as_real"],
            "fake_classified_as_fake": after_metrics["fake_classified_as_fake"],
        },
    ]

    with open(results_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device(DEVICE_NAME)
    train_loader = create_loader(DATA_ROOT / "train", BATCH_SIZE, NUM_WORKERS, shuffle=True)
    val_loader = create_loader(DATA_ROOT / "validation", BATCH_SIZE, NUM_WORKERS, shuffle=False)
    test_loader = create_loader(DATA_ROOT / "test", BATCH_SIZE, NUM_WORKERS, shuffle=False)
    model = load_model(MODEL_NAME, CHECKPOINT_PATH, device)
    loss_fn = nn.BCEWithLogitsLoss()

    print("Evaluating before transfer learning...")
    before_val_metrics = evaluate_model(model, val_loader, loss_fn, device, SEED)
    before_test_metrics = evaluate_model(model, test_loader, loss_fn, device, SEED + 1)
    best_head_state = deepcopy(model.attention_head.state_dict())
    best_val_loss = before_val_metrics["loss"]
    epochs_without_improvement = 0
    optimizer = torch.optim.Adam(model.attention_head.parameters(), lr=LEARNING_RATE)

    print("Training transfer learning model...")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate_model(model, val_loader, loss_fn, device, SEED)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_head_state = deepcopy(model.attention_head.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

    model.attention_head.load_state_dict(best_head_state)
    torch.save(best_head_state, OUTPUT_DIR / "best_attention_head.pth")
    print("Evaluating after transfer learning...")
    after_test_metrics = evaluate_model(model, test_loader, loss_fn, device, SEED + 1)
    write_results_csv(RESULTS_CSV_PATH, before_test_metrics, after_test_metrics)
    print("Done. Saved results.csv and best_attention_head.pth")

if __name__ == "__main__":
    main()
