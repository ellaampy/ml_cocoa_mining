import torch
import numpy as np
import os
import albumentations as A

def save_predictions(model, dataloader, out_dir, device='cuda'):

    model.eval()
    # Store predictions in a list
    all_preds = []
    all_labels = []
    all_ids = []

    # Iterate over the validation or test set
    with torch.no_grad():
        for inputs, labels, ids in dataloader: 
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            # Revert padding
            pad_size = (224 - 128) // 2
            outputs = outputs[:, :, pad_size:pad_size+128, pad_size:pad_size+128]
            labels = labels[:, pad_size:pad_size+128, pad_size:pad_size+128]

            # Convert model outputs to class predictions
            _, preds = torch.max(outputs, 1)

            # Save predictions and labels
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_ids.append(ids)

    # Convert lists to numpy arrays
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_ids = np.concatenate(all_ids, axis=0)

    # Save predictions and labels to a file
    np.save(os.path.join(out_dir, 'predictions.npy'), all_preds)
    np.save(os.path.join(out_dir, 'labels.npy'), all_labels)
    np.save(os.path.join(out_dir, 'ids.npy'), all_ids)



def get_transform(transform_type):

    if transform_type == 'simple':
        transformed = A.Compose([
            
            A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, value=0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.CropAndPad(px=60, p=0.2),
            A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), p=0.5)
        ])

    if transform_type == 'simple_norm':
        transformed = A.Compose([
            
            A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, value=0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.CropAndPad(px=60, p=0.2),
            A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(.229, 0.224, 0.225))
        ])

    elif transform_type == 'colour':
        transformed = A.Compose([
        
            A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, value=0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.CropAndPad(px=60, p=0.2),
            A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), p=0.5),
            A.ChannelShuffle(p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, always_apply=None, p=0.2),
            A.RandomBrightnessContrast(brightness_limit=(-0.05, 0.05), contrast_limit=(-0.05, 0.05), p=0.2)

        ])
    
    # elif transform_type == 'cutout':
    #     transformed = A.Compose([
            
    #         A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, value=0),
    #         A.HorizontalFlip(p=0.5),
    #         A.VerticalFlip(p=0.5),
    # A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), p=0.5),
    #         A.RandomResizedCrop(size=(224,224), p=0.2),
    #         A.Cutout(num_holes=8, max_h_size=16, max_w_size=16, fill_value=0, p=0.3)
    #     ])

    elif transform_type == 'mixed':
        transformed = A.Compose([
            
            A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, value=0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.CropAndPad(px=60, p=0.2),
            A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), p=0.5),
            A.ChannelShuffle(p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, always_apply=None, p=0.2),
            A.RandomBrightnessContrast(brightness_limit=(-0.05, 0.05), contrast_limit=(-0.05, 0.05), p=0.2),
            A.Cutout(num_holes=8, max_h_size=16, max_w_size=16, fill_value=0, p=0.1)

        ])
        
    elif transform_type =='validation':
        transformed = A.Compose([
            A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, value=0)])

    elif transform_type =='validation_norm':
        transformed = A.Compose([
            A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, value=0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(.229, 0.224, 0.225))])
        
    return transformed




from typing import Optional, List

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
def soft_dice_score(
    output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0, eps: float = 1e-7, dims=None
) -> torch.Tensor:
    """

    :param output:
    :param target:
    :param smooth:
    :param eps:
    :return:

    Shape:
        - Input: :math:`(N, NC, *)` where :math:`*` means any number
            of additional dimensions
        - Target: :math:`(N, NC, *)`, same shape as the input
        - Output: scalar.

    """
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return dice_score

class DiceLoss(_Loss):
    def __init__(
        self,
        mode: str,
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
    ):
        """Dice loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            classes:  List of classes that contribute in loss computation. By default, all channels are included.
            log_loss: If True, loss computed as `- log(dice_coeff)`, otherwise `1 - dice_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient (a)
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        super(DiceLoss, self).__init__()
        self.mode = mode
        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == 'multiclass':
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)


        if self.mode == 'multiclass':
            y_pred = y_pred.reshape(bs, num_classes, -1)
            y_true = y_true.reshape(bs, -1)
            

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)

                y_true = F.one_hot(
                    (y_true * mask).to(torch.long), num_classes
                )  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)  # N, C, H*W
            else:
                y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1)  # N, C, H*W


        scores = self.compute_score(
            y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims
        )

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        return self.aggregate_loss(loss)

    def aggregate_loss(self, loss):
        return loss.mean()

    def compute_score(
        self, output, target, smooth=0.0, eps=1e-7, dims=None
    ) -> torch.Tensor:
        return soft_dice_score(output, target, smooth, eps, dims)