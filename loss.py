import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def bce_loss(y_real, y_pred):
    return torch.mean(y_pred - y_real*y_pred + torch.log(1 + torch.exp(-y_pred)))

def dice_loss(y_real, y_pred):
    # encode y_pred as a binary mask with gradient
    y_pred = torch.sigmoid(y_pred)
    
    smooth = 1e-10
    y_pred = y_pred.contiguous().view(-1)
    y_real = y_real.contiguous().view(-1)
    intersection = (y_pred * y_real).sum()
    A_sum = torch.sum(y_pred * y_pred)
    B_sum = torch.sum(y_real * y_real)
    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )
    #dice_loss = 1 - torch.mean(2*y_real*y_pred +1) / (torch.mean(y_real + y_pred)+1)
    #return dice_loss

# def focal_loss(y_real, y_pred):
#     """
#     Compute binary focal loss for a batch of data.
    
#     Args:
#         y_real (Tensor): Ground truth labels (batch_size, height, width) with values in {0, 1}.
#         y_pred (Tensor): Predicted logits (batch_size, height, width) before applying the sigmoid activation.
#         alpha (float): Weighting factor for class imbalance (default: 0.25).
#         gamma (float): Focusing parameter for hard-to-classify examples (default: 2.0).
#         eps (float): Small epsilon to avoid log(0).
    
#     Returns:
#         Tensor: Scalar focal loss value.
#     """
#     alpha=0.25
#     gamma=2.0
#     eps=1e-8
#     # Apply sigmoid to the predicted logits to get probabilities
#     y_pred = torch.sigmoid(y_pred)

#     # Flatten the tensors to handle batches
#     y_pred = y_pred.contiguous().view(-1)
#     y_real = y_real.contiguous().view(-1)

#     # Compute the focal loss components
#     p_t = y_real * y_pred + (1 - y_real) * (1 - y_pred)  # p_t = p when y_real == 1, and 1-p when y_real == 0
#     alpha_t = y_real * alpha + (1 - y_real) * (1 - alpha)  # alpha_t = alpha when y_real == 1, and 1-alpha when y_real == 0

#     # Focal loss with focusing parameter gamma
#     focal_weight = alpha_t * (1 - p_t) ** gamma

#     # Add a small epsilon to avoid log(0)
#     loss = -focal_weight * torch.log(p_t + eps)

#     # Return mean loss over the batch
#     return loss.mean()

def focal_loss(y_real, y_pred, alpha=0.25, gamma=2.0, eps=1e-8):
    """
    Binary focal loss for imbalanced datasets.
    Args:
        y_real (Tensor): Ground truth labels (0 or 1).
        y_pred (Tensor): Raw logits (not probabilities).
        alpha (float): Weighting factor for class imbalance.
        gamma (float): Focusing parameter for hard-to-classify examples.
        eps (float): Small epsilon to avoid numerical instability.
    Returns:
        Tensor: Scalar focal loss.
    """
    # Apply sigmoid to logits
    y_pred = torch.sigmoid(y_pred)

    # Flatten tensors to handle batches
    y_pred = y_pred.contiguous().view(-1)
    y_real = y_real.contiguous().view(-1)

    # Clamp p_t to avoid log(0)
    p_t = torch.clamp(y_real * y_pred + (1 - y_real) * (1 - y_pred), min=eps, max=1.0)
    
    # Compute alpha_t for class weighting
    alpha_t = y_real * alpha + (1 - y_real) * (1 - alpha)

    # Compute focal weight
    focal_weight = alpha_t * (1 - p_t) ** gamma

    # Compute final loss
    loss = -focal_weight * torch.log(p_t)
    return loss.mean()


def bce_total_variation(y_real, y_pred):
    return bce_loss(y_real, y_pred) + 0.15 * (torch.mean(torch.sigmoid(y_pred[:,:,1:,:]) - torch.sigmoid(y_pred[:,:,:-1,:])) + torch.mean(torch.sigmoid(y_pred[:,:,:,1:]) - torch.sigmoid(y_pred[:,:,:,:-1])))

def weighted_bce_loss(y_real, y_pred):
    # get weights as inverse class frequency
    mean = torch.mean(y_real, dim=(2,3))
    class_weights = torch.zeros_like(y_real)
    class_weights[0,0,:,:] = torch.where(y_real[0,0,:,:]>0, 1 - mean[0,0], mean[0,0])
    class_weights[1,0,:,:] = torch.where(y_real[1,0,:,:]>0, 1 - mean[1,0], mean[1,0])
    
    return nn.BCEWithLogitsLoss(pos_weight=class_weights)(y_pred, y_real)