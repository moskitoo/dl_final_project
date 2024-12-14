import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# pip install torchsummary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
#from torchsummary import summary
import torch.optim as optim
from torchvision.transforms import v2
from time import time
import wandb
from model import BaseUnet, BaseUnet3D
from dataset import BrightfieldMicroscopyDataset
from early_stopping import EarlyStopping
from arguments import parse_args, parse_args_3dunet
from evaluation_metrics import dice_overlap, intersection_over_union, accuracy, sensitivity, specificity
from loss import dice_loss, weighted_bce_loss, focal_loss
from data_processing_tools import remove_repeating_pattern

wandb.login(key=os.environ.get('WANDB_API_KEY'))


def get_dataloader(sample_size, batch_size):
 
    image_root = '/zhome/70/5/14854/nobackup/deeplearningf24/forcebiology/data/brightfield'
    mask_root = '/zhome/70/5/14854/nobackup/deeplearningf24/forcebiology/data/masks'

    # image_root = 'data/brightfield'
    # mask_root = 'data/masks'

    transform_train = v2.Compose([
        v2.Resize((sample_size, sample_size)),
        v2.RandomRotation(degrees=15),
        v2.RandomHorizontalFlip(p=0.3),
        v2.ToDtype(torch.float32, scale=True),
        v2.ToTensor(),
    ])

    transform_val = v2.Compose([
        v2.Resize((sample_size, sample_size)),
        v2.ToDtype(torch.float32, scale=True),
        v2.ToTensor(),
    ])

    brightfield_train_datatset = BrightfieldMicroscopyDataset(root_dir_images=image_root, root_dir_labels=mask_root, train=True, transform=transform_train)
    brightfield_val_datatset = BrightfieldMicroscopyDataset(root_dir_images=image_root, root_dir_labels=mask_root, train=False, validation=True, transform=transform_val)
    brightfield_test_datatset = BrightfieldMicroscopyDataset(root_dir_images=image_root, root_dir_labels=mask_root, train=False, validation=False, transform=transform_val)

    brightfield_loader_train = DataLoader(brightfield_train_datatset,  batch_size=batch_size, shuffle=True)
    brightfield_loader_val = DataLoader(brightfield_val_datatset,  batch_size=batch_size, shuffle=True)
    brightfield_loader_test = DataLoader(brightfield_test_datatset,  batch_size=1, shuffle=False)

    return brightfield_loader_train, brightfield_loader_val, brightfield_loader_test

def checkpoint_model(model, optimiser, epoch, path='model.pth'):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimiser_state_dict': optimiser.state_dict(),
            }, path)

def save_model(model, path='model.pth'):
    torch.save(model.state_dict(), path)

def train_model(model, train_loader, val_loader, test_loader, optimiser, lr_scheduler, criterion, device, args, early_stopping, num_epochs=10):
    # Initialize W&B run
    wandb.init(
        project=args.project_name,         
        entity="hndrkjs-danmarks-tekniske-universitet-dtu",           
        config={
            "epochs": num_epochs,
            "learning_rate": optimiser.param_groups[0]['lr'],
            "batch_size": args.batch_size,
            "model_name": args.model_name,
        }
    )
    
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            images, labels = data

            # remove repeating pattern
            for i in range(images.shape[0]):
                images[i] = torch.tensor(remove_repeating_pattern(images[i].numpy()))

            images, labels = images.to(device), labels.to(device)

            if args.train_3d:
                images = images.unsqueeze(2)
                labels = labels.unsqueeze(1)
         
            optimiser.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimiser.step()
            running_loss += loss.item()

        # Calculate average training loss for the epoch
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation step
        model.eval()
        val_loss = 0.0
        dice = 0
        iou = 0
        acc = 0
        sens = 0
        spec = 0
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                # remove repeating pattern
                for i in range(images.shape[0]):
                    images[i] = torch.tensor(remove_repeating_pattern(images[i].numpy()))
                
                images, labels = images.to(device), labels.to(device)

                if args.train_3d:
                    images = images.unsqueeze(2)
                    labels = labels.unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()

                Y_pred = (outputs > 0.45).float()
                dice += dice_overlap(Y_pred, labels)
                iou += intersection_over_union(Y_pred, labels)
                acc += accuracy(Y_pred, labels)
                sens += sensitivity(Y_pred, labels)
                spec += specificity(Y_pred, labels)

                # concatenate y_batch and y_pred to log
                image_array = np.concatenate([labels[0].cpu().numpy(), Y_pred[0].detach().cpu().numpy()], axis=1)

                images = wandb.Image(image_array, caption="Top: Input, Bottom: Output")
            
            dice /= len(val_loader)
            iou /= len(val_loader)
            acc /= len(val_loader)
            sens /= len(val_loader)
            spec /= len(val_loader)

        avg_val_loss = val_loss / len(val_loader)

        # Adjust learning rate after each epoch
        if args.lr_scheduler:
            lr_scheduler.step()

        # Log metrics to W&B
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": optimiser.param_groups[0]['lr'],
            "Dice": dice,
            "IoU": iou,
            "Accuracy": acc,
            "Sensitivity": sens,
            "Specificity": spec
        })
        wandb.log({"Predicted segmentation": images})

        # Save model checkpoint to W&B
        # if epoch % 10 == 0:
        #     #wandb.save('checkpoint_model_{}_{}.pth'.format(args.project_name, epoch))
        #     checkpoint_model(model, optimiser, epoch, path='checkpoint_model_{}_{}.pth'.format(args.project_name, epoch))

        early_stopping(avg_val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    # Save the final model and upload to W&B
    # final_model_path = 'final_model_classification{}.pth'.format(args.model_name)
    # save_model(model, path=final_model_path)
    
    # test the model
    model.eval()
    test_loss = 0
    dice = 0
    iou = 0
    acc = 0
    sens = 0
    spec = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # remove repeating pattern
            for i in range(images.shape[0]):
                images[i] = torch.tensor(remove_repeating_pattern(images[i].numpy()))
            
            images, labels = images.to(device), labels.to(device)

            if args.train_3d:
                images = images.unsqueeze(2)
                labels = labels.unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels.float())
            test_loss += loss.item()

            Y_pred = (outputs > 0.45).float()
            dice += dice_overlap(Y_pred, labels)
            iou += intersection_over_union(Y_pred, labels)
            acc += accuracy(Y_pred, labels)
            sens += sensitivity(Y_pred, labels)
            spec += specificity(Y_pred, labels)
        
        dice /= len(test_loader)
        iou /= len(test_loader)
        acc /= len(test_loader)
        sens /= len(test_loader)
        spec /= len(test_loader)
    
    wandb.log({
        "test_loss": test_loss / len(test_loader),
        "Dice Test": dice,
        "IoU Test": iou,
        "Accuracy Test": acc,
        "Sensitivity Test": sens,
        "Specificity Test": spec
    })

    # Finish the W&B run
    wandb.finish()


if __name__ == '__main__':

    args = parse_args_3dunet()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BaseUnet3D(num_inputs=11)

    train_loader, val_loader, test_loader = get_dataloader(args.sample_size, args.batch_size)

    criterion = nn.BCEWithLogitsLoss()

    # Initialize optimiser and learning rate scheduler
    if args.optimiser == 'adam':
        optimiser = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimiser == 'sgd':
        optimiser = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError('optimiser should be either adam or sgd')
    
    lr_scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=args.step_size, gamma=args.gamma)

    early_stopping = EarlyStopping(patience=args.patience, delta=args.delta, verbose=False, path='early_stopping_model{}.pth'.format(args.project_name))

    # Train the model
    train_model(model, train_loader, val_loader, test_loader,
                num_epochs=args.num_epochs, 
                optimiser=optimiser, lr_scheduler=lr_scheduler, 
                criterion=criterion, device=device, args=args, 
                early_stopping=early_stopping)