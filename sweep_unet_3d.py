import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torch

# pip install torchsummary
import torch
import torch.nn as nn
import torchvision
torchvision.disable_beta_transforms_warning()
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.transforms import v2
import wandb
from model import BaseUnet, BaseUnet3D
from dataset import BrightfieldMicroscopyDataset
from evaluation_metrics import dice_overlap, intersection_over_union, accuracy, sensitivity, specificity

wandb.login(key=os.environ.get('WANDB_API_KEY'))

sweep_config = {
    'method': 'random'
    }

metric = {
    'name': 'test_loss',
    'goal': 'minimize'   
    }

sweep_config['metric'] = metric

parameters_dict = {
    'optimizer': {
        'values': ['adam', 'sgd']
        },
    'rotation': {
        'values': [0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
        },
    'horizontal_flip_prob': {
        'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        },
    'photo_distort_prob': {
        'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        },
    'image_size': {
        'values': [512]
        },
    'learning_rate': {
        # a flat distribution between 0 and 0.001
        'distribution': 'uniform',
        'min': 0,
        'max': 0.001
      },
    'weight_decay': {
        # a flat distribution between 0 and 0.0001
        'distribution': 'uniform',
        'min': 0,
        'max': 0.0001
      },
    'batch_size': {
        # integers between 1 and 8
        # with evenly-distributed logarithms 
        'distribution': 'q_log_uniform_values',
        'q': 2,
        'min': 8,
        'max': 18,
      },
    'epochs': {
        'value': 4}
    }

sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-3DCNN-brightfield_segmentation")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_dataset_segmentation(batch_size, image_size, random_rotation_degrees, horizontal_flip_prob, photo_distort_prob):
    
    image_root = '/zhome/70/5/14854/nobackup/deeplearningf24/forcebiology/data/brightfield'
    mask_root = '/zhome/70/5/14854/nobackup/deeplearningf24/forcebiology/data/masks'

    # image_root = 'data/brightfield'
    # mask_root = 'data/masks'

    transform_train = v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.RandomRotation(random_rotation_degrees),
        v2.RandomHorizontalFlip(horizontal_flip_prob),
        v2.ToDtype(torch.float32, scale=True),
        v2.ToTensor()
    ])

    transform_val = v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.ToDtype(torch.float32, scale=True),
        v2.ToTensor()
    ])

    brightfield_train_datatset = BrightfieldMicroscopyDataset(root_dir_images=image_root, root_dir_labels=mask_root, train=True, transform=transform_train)
    brightfield_val_datatset = BrightfieldMicroscopyDataset(root_dir_images=image_root, root_dir_labels=mask_root, train=False, validation=True, transform=transform_val)

    brightfield_loader_train = DataLoader(brightfield_train_datatset,  batch_size=batch_size, shuffle=True)
    brightfield_loader_val = DataLoader(brightfield_val_datatset,  batch_size=batch_size, shuffle=True)

    return brightfield_loader_train, brightfield_loader_val

def build_optimizer(network, optimizer, learning_rate, weight_decay):
    if optimizer == "sgd":

        optimiser = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

    elif optimizer == "adam":
        
        optimiser = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return optimiser

def train_epoch(network, train_loader, val_loader, optimizer, loss_fn):
    cumu_loss = 0
    cumu_loss_test = 0
    total = 0
    correct = 0
    acc = 0
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        data = data.unsqueeze(2)
        target = target.unsqueeze(1)

        # ➡ Forward pass
        pred = network(data)
        loss = loss_fn(pred, target.float())
        cumu_loss += loss.item()

        # ⬅ Backward pass + weight update
        loss.backward()
        optimizer.step()

        # Evaluate model
    network.eval()
    val_loss = 0
    dice = 0
    iou = 0
    acc = 0
    sens = 0
    spec = 0

    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            images = images.unsqueeze(2)
            labels = labels.unsqueeze(1)

            outputs = network(images)
            loss_val = loss_fn(outputs, labels.float())
            val_loss += loss.item()

            Y_pred = (outputs > 0.45).float()
            dice += dice_overlap(Y_pred, labels)
            iou += intersection_over_union(Y_pred, labels)
            acc += accuracy(Y_pred, labels)
            sens += sensitivity(Y_pred, labels)
            spec += specificity(Y_pred, labels)
            
        dice /= len(val_loader)
        iou /= len(val_loader)
        acc /= len(val_loader)
        sens /= len(val_loader)
        spec /= len(val_loader)  
    network.train()  

    return (cumu_loss / len(train_loader)), (val_loss / len(val_loader)), acc, dice

LOSS_FN = nn.BCEWithLogitsLoss()

def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        train_loader, val_loader = build_dataset_segmentation(config.batch_size, config.image_size, config.rotation, config.horizontal_flip_prob, config.photo_distort_prob)

        network = BaseUnet3D(num_inputs=11)
        network.to(device)

        optimizer = build_optimizer(network, config.optimizer, config.learning_rate, config.weight_decay)

        for epoch in range(config.epochs):

            avg_loss, avg_test_loss, accuracy, dice = train_epoch(network, train_loader, val_loader, optimizer, LOSS_FN)
            wandb.log({"loss": avg_loss, "test_loss":  avg_test_loss , "accuracy": accuracy, "dice": dice, "epoch": epoch}) 

if __name__ == '__main__':

    wandb.agent(sweep_id, train, count=100)