import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import BrightfieldMicroscopyDataset
from unet import UNet
from torchvision.transforms import v2
import os
import logging


DATA_ROOT = "/zhome/70/5/14854/nobackup/deeplearningf24/forcebiology/data"  
LOG_DIR = "/zhome/68/f/213210/deep_learning_final_project/output_logs"    
# MODEL_SAVE_PATH = "/zhome/68/f/213210/deep_learning_final_project/models/unet_model.pth" 
LOG_FILE = os.path.join(LOG_DIR, "training_log.log")
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
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    correct_pixels = 0
    total_pixels = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            val_loss += loss.item()
            predictions = torch.sigmoid(outputs) > 0.5  # Binary threshold
            correct_pixels += (predictions == labels.unsqueeze(1)).sum().item()
            total_pixels += labels.numel()

    val_loss /= len(val_loader)
    val_accuracy = correct_pixels / total_pixels * 100

    # Log epoch results
    logging.info(
        f"Epoch {epoch+1}/{EPOCHS} - Training Loss: {train_loss:.4f}, "
        f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%"
    )

print(f"Training complete.")


# Save the model
# torch.save(model.state_dict(), MODEL_SAVE_PATH)
# print(f"Model saved to {MODEL_SAVE_PATH}")

