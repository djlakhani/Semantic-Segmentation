import numpy as np
import torch

def iou(pred, target, n_classes = 21):
    """
    Calculate the Intersection over Union (IoU) for predictions.

    Args:
        pred (tensor): Predicted output from the model.
        target (tensor): Ground truth labels.
        n_classes (int, optional): Number of classes. Default is 21.

    Returns:
        float: Mean IoU across all classes.
    """

    batch_size = pred.shape[0]
    iou_sum = torch.zeros(batch_size, device=pred.device)
    class_count = torch.zeros(batch_size, device=pred.device)

    for cls in range(n_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        intersection = (pred_cls & target_cls).sum(dim=(1, 2))
        union = (pred_cls | target_cls).sum(dim=(1, 2))

        cls_present = union != 0
        zero_mask = union == 0
        union[zero_mask] = 1
        iou = intersection / union

        iou_sum = iou_sum + iou
        class_count = class_count + cls_present

    class_count[class_count == 0] = 1
    return iou_sum / class_count


def pixel_acc(pred, target):
    """
    Calculate pixel-wise accuracy between predictions and targets.

    Args:
        pred (tensor): Predicted output from the model.
        target (tensor): Ground truth labels.

    Returns:
        float: Pixel-wise accuracy.
    """

    outputs = (pred == target)
    outputs_sum = torch.sum(outputs, (1, 2))
    totals = (target == target)
    totals_sum = torch.sum(totals, (1, 2))
   
    return outputs_sum / totals_sum