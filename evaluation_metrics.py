import torch
from torch import nn

def specificity(pred, target):
    # pred and target are of shape (batch_size, 1, height, width)

    smooth = 1e-10
    pred = pred.view(-1)
    target = target.view(-1)
    true_negatives = torch.sum((pred == 0) & (target == 0))
    actual_negatives = torch.sum(target == 0)
    return (true_negatives.float() + smooth) / (actual_negatives + smooth)

def sensitivity(pred, target):
    # pred and target are of shape (batch_size, 1, height, width)
    smooth = 1e-10
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = torch.sum(pred*target)
    return (intersection + smooth) / (torch.sum(target) + smooth)

def accuracy(pred, target):
    # pred and taeget are of shape (batch_size, 1, height, width)
    pred = pred.view(-1)
    target = target.view(-1)
    correct = torch.sum(pred == target)
    total = torch.numel(pred)
    return correct.float() / total

def intersection_over_union(pred, target):
    # pred and target are of shape (batch_size, 1, height, width)
    smooth = 1e-10
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def dice_overlap(pred, target):
    # pred and target are of shape (batch_size, 1, height, width)
    smooth = 1e-10
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)