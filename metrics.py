import torch
import torch.nn.functional as F


def dice_coefficient(preds, targets, num_classes=4, eps=1e-6):
    """Mean Dice across classes. preds/targets: (B,H,W) class-index tensors."""
    scores = []
    for c in range(num_classes):
        pred_c = (preds == c).float()
        target_c = (targets == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        scores.append(((2 * intersection + eps) / (union + eps)).item())
    return sum(scores) / num_classes, scores


def iou_score(preds, targets, num_classes=4, eps=1e-6):
    scores = []
    for c in range(num_classes):
        pred_c = (preds == c).float()
        target_c = (targets == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - intersection
        scores.append(((intersection + eps) / (union + eps)).item())
    return sum(scores) / num_classes, scores


def pixel_accuracy(preds, targets):
    correct = (preds == targets).float().sum()
    return (correct / torch.numel(targets)).item()


def soft_dice_loss(logits, targets, num_classes=4, eps=1e-6):
    """Differentiable Dice loss for backprop."""
    probs = F.softmax(logits, dim=1)
    targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probs * targets_onehot).sum(dims)
    union = probs.sum(dims) + targets_onehot.sum(dims)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()