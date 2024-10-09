import hydra
from omegaconf import DictConfig, OmegaConf
import random
from dataset import ImageMaskDataset
from utils import save_predictions, get_transform, DiceLoss
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import TverskyLoss, FocalLoss
from torch.utils.data import DataLoader
import albumentations as A
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter


@hydra.main(config_path=".", config_name="config", version_base="1.1")

def main(cfg: DictConfig):

    # create results dir and save config
    os.makedirs(cfg.training.results_path, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.training.results_path, 'config.yaml'))

    # set seed for reproducability
    np.random.seed(cfg.training.seed)
    random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    device = torch.device(cfg.training.device)

    # ================ DATALOADER ======================
    train_dataset = ImageMaskDataset(cfg.dataset.data_dir, cfg.dataset.split_path, 'train', cfg.dataset.feature_idx, 
                               cfg.dataset.classification, 0.65, get_transform(cfg.dataset.transform))
    
    val_dataset = ImageMaskDataset(cfg.dataset.data_dir, cfg.dataset.split_path, 'test', cfg.dataset.feature_idx, 
                               cfg.dataset.classification, 0.65, get_transform('validation'))

    # dataLoader
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers)
    
    # ================ ======================
    if cfg.model.name == 'unet':
        model = smp.Unet(cfg.model.backbone, encoder_weights=cfg.model.encoder_weights, classes=cfg.model.num_classes)
    elif cfg.model.name == 'unet_scratch':
        model = smp.Unet(cfg.model.backbone, encoder_weights=None, classes=cfg.model.num_classes)
    elif cfg.model.name == 'unet_p':
        model = smp.UnetPlusPlus(cfg.model.backbone, encoder_weights=cfg.model.encoder_weights, classes=cfg.model.num_classes)
    elif cfg.model.name == 'deeplab':
        model = smp.DeepLabV3(cfg.model.backbone, encoder_weights=cfg.model.encoder_weights, classes=cfg.model.num_classes)

    # print(model)

    # freeze weights
    if cfg.model.unfreeze is not None:
 
        if cfg.model.unfreeze == 'decoder':
            for param in model.encoder.parameters():
                param.requires_grad = False

        elif cfg.model.unfreeze == 'segmentation_head':
            for param in model.parameters():
                param.requires_grad = False

            for param in model.segmentation_head.parameters():
                param.requires_grad = True

    # adapt first conv layer to extentend channels
    if len(cfg.dataset.feature_idx) > 3:

        original_conv1 = model.encoder.conv1
        new_conv1 = nn.Conv2d(
            in_channels=len(cfg.dataset.feature_idx), 
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None
        )
        # Initialize new weights: Copy pre-trained weights for the first 3 channels
        with torch.no_grad():
            new_conv1.weight[:, :3] = original_conv1.weight
            if new_conv1.weight.shape[1] > 3:
                # Initialize remaining channels, e.g., using zeros or random initialization
                new_conv1.weight[:, 3:] = torch.mean(original_conv1.weight, dim=1, keepdim=True)

        model.encoder.conv1 = new_conv1


    model = model.to(cfg.training.device)

    if cfg.training.loss == 'focal':
        criterion = FocalLoss(mode='multiclass')
    elif cfg.training.loss == 'dice':
        criterion = DiceLoss(mode='multiclass')
    elif cfg.training.loss == 'tversky':
        criterion = TverskyLoss(mode='multiclass')


    optimizer_class = getattr(optim, cfg.training.optimizer.capitalize())
    optimizer = optimizer_class(model.parameters(), lr=cfg.training.lr)
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.1)

    # train block
    best_iou = 0.0
    best_epoch = 0

    dataloaders = {'train':train_loader, 'valid': val_loader}
    dataset_sizes = {'train':len(train_dataset), 'valid': len(val_dataset)}


    # Initialize TensorBoard SummaryWriter
    writer_train = SummaryWriter(log_dir=os.path.join(cfg.training.results_path, 'log', 'train'))
    writer_val = SummaryWriter(log_dir=os.path.join(cfg.training.results_path, 'log', 'valid'))

    for epoch in range(cfg.training.epochs):
        print(f'Epoch {epoch}/{cfg.training.epochs - 1}')
        print('-' * 40)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels,_ in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)

                    # revert padding (original image:128x128)
                    pad_size = (224 - 128) // 2
                    outputs = outputs[:, :, pad_size:pad_size+128, pad_size:pad_size+128]
                    labels = labels[:, pad_size:pad_size+128, pad_size:pad_size+128]

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'validation':
                exp_lr_scheduler.step(loss)


            ############ METRICS
            tp, fp, fn, tn = smp.metrics.get_stats(preds, labels, mode='multiclass', 
                                                   num_classes=cfg.model.num_classes)
            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
            # f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
            accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
            # recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")

            epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_corrects.double() / (dataset_sizes[phase]* 128 * 128)
            print(f'{phase} Loss: {epoch_loss:.4f}   mIoU: {iou_score:.4f}   F1: {f1_score:.4f}   Acc: {accuracy:.4f}')


            # write to tensorboard
            if phase == 'train':
                writer_train.add_scalar('Loss', epoch_loss, epoch)
                writer_train.add_scalar('mIoU', iou_score, epoch)
            else:
                writer_val.add_scalar('Loss', epoch_loss, epoch)
                writer_val.add_scalar('mIoU', iou_score, epoch)

            # deep copy the model
            if phase == 'valid' and iou_score >= best_iou:
                best_iou = iou_score
                best_epoch = epoch
                torch.save({'state_dict': model.state_dict()}, os.path.join(cfg.training.results_path, 'model.pth.tar'))

    # save predictions on validation
    model.load_state_dict(torch.load(os.path.join(cfg.training.results_path, 'model.pth.tar'))['state_dict'])
    save_predictions(model, val_loader, cfg.training.results_path, device='cuda')

    print()
    print(f'Best val mIoU: {best_iou:4f} occured at {best_epoch}')

    writer_train.close()
    writer_val.close()


if __name__ == "__main__":
    print(torch.__version__)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False
    torch.cuda.empty_cache()
    main()