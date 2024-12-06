import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import BrightfieldMicroscopyDataset
from unet import UNet
from torchvision.transforms import v2
import os
import logging
from evaluation_metrics import specificity, sensitivity, accuracy, intersection_over_union, dice_overlap


DATA_ROOT = "/zhome/70/5/14854/nobackup/deeplearningf24/forcebiology/data"  
LOG_DIR = "/zhome/68/f/213210/deep_learning_final_project/output_logs"    
MODEL_SAVE_PATH = "/zhome/68/f/213210/deep_learning_final_project/models/unet_model_6_1155.pth" 
LOG_FILE = os.path.join(LOG_DIR, "training_log_with_metrics_6_1155.log")
os.makedirs(LOG_DIR, exist_ok=True)


# Configure Logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 10

# Transforms
train_transform = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize(512),
    # v2.RandomHorizontalFlip(p=0.5),
])

val_transform = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize(512),
])

# Datasets and DataLoaders
train_dataset = BrightfieldMicroscopyDataset(
    root_dir_images=f"{DATA_ROOT}/brightfield",
    root_dir_labels=f"{DATA_ROOT}/masks",
    train=True,
    validation=False,
    transform=train_transform
)

val_dataset = BrightfieldMicroscopyDataset(
    root_dir_images=f"{DATA_ROOT}/brightfield",
    root_dir_labels=f"{DATA_ROOT}/masks",
    train=False,
    validation=True,
    transform=val_transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = UNet().cuda()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    # Training Phase
    model.train()
    train_loss = 0
    correct_train_pixels = 0
    total_train_pixels = 0

    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # Calculate training accuracy
        predictions = torch.sigmoid(outputs) > 0.5
        correct_train_pixels += (predictions == labels.unsqueeze(1)).sum().item()
        total_train_pixels += labels.numel()

    train_loss /= len(train_loader)
    train_accuracy = correct_train_pixels / total_train_pixels * 100

    # Validation Phase
    model.eval()
    val_loss = 0
    correct_val_pixels = 0
    total_val_pixels = 0
    total_specificity = 0
    total_sensitivity = 0
    total_iou = 0
    total_dice = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels.unsqueeze(1).float())
            val_loss += loss.item()

            # Compute predictions
            predictions = torch.sigmoid(outputs) > 0.5

            # Calculate metrics
            total_specificity += specificity(predictions, labels).item()
            total_sensitivity += sensitivity(predictions, labels).item()
            total_iou += intersection_over_union(predictions, labels).item()
            total_dice += dice_overlap(predictions, labels).item()
            correct_val_pixels += (predictions == labels.unsqueeze(1)).sum().item()
            total_val_pixels += labels.numel()

    # Average Metrics
    val_loss /= len(val_loader)
    val_accuracy = correct_val_pixels / total_val_pixels * 100
    avg_specificity = total_specificity / len(val_loader)
    avg_sensitivity = total_sensitivity / len(val_loader)
    avg_iou = total_iou / len(val_loader)
    avg_dice = total_dice / len(val_loader)

    # Log Metrics
    logging.info(
        f"Epoch {epoch+1}/{EPOCHS} - "
        f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, "
        f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%, "
        f"Specificity: {avg_specificity:.4f}, Sensitivity: {avg_sensitivity:.4f}, "
        f"IoU: {avg_iou:.4f}, Dice Coefficient: {avg_dice:.4f}")

print(f"Training complete.")
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

