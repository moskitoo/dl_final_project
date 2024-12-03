import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import BrightfieldMicroscopyDataset
from unet import UNet
from torchvision.transforms import v2
import os

DATA_ROOT = "/zhome/70/5/14854/nobackup/deeplearningf24/forcebiology/data"  
LOG_DIR = "/zhome/68/f/213210/deep_learning_final_project/output_logs"    
# MODEL_SAVE_PATH = "/zhome/68/f/213210/deep_learning_final_project/models/unet_model.pth" 
os.makedirs(LOG_DIR, exist_ok=True)

BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 25

train_transform = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize(512),
    v2.RandomHorizontalFlip(p=0.5),
])

val_transform = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize(512),
])

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

# Model, Loss, and Optimizer
model = UNet()
model = model.cuda()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training and Validation Loop
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
    print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {train_loss:.4f}")
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            val_loss += loss.item()
    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Validation Loss: {val_loss:.4f}")

# Save the model
# torch.save(model.state_dict(), MODEL_SAVE_PATH)
# print(f"Model saved to {MODEL_SAVE_PATH}")

