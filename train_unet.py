from dataset import ImageMaskDataset
from utils import save_predictions
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, TverskyLoss, FocalLoss
# import pytorch_lightning as pl
# from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import albumentations as A
import os
from torch.utils.tensorboard import SummaryWriter


def train_model(data_dir, split_path, feature_idx, classification, backbone, encoder_weights, num_classes, out_dir=None):


    # create output dir
    os.makedirs(out_dir, exist_ok=True)
    # https://www.kaggle.com/code/whurobin/data-augmentation-test-with-albumentations
    transform = A.Compose([
        
        A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, value=0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.RandomResizedCrop(size=128, p=0.2),
        # A.GaussianBlur(blur_limit=3, p=1),
        # A.RandomBrightness(limit=0.2, p=0.1),
        # A.Cutout(num_holes=8, max_h_size=16, max_w_size=16, fill_value=0, p=0.1),
        # A.ChannelShuffle(p=0.1)

    ])

    transform_v = A.Compose([
         A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, value=0),
    ])

    train_dataset = ImageMaskDataset(data_dir, split_path, 'train', feature_idx, 
                               classification, 0.65, transform)
    
    val_dataset = ImageMaskDataset(data_dir, split_path, 'test', feature_idx, 
                               classification, 0.65, transform_v)

    # # Define the DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8)


    device = 'cuda'
    model = smp.Unet(backbone, encoder_weights=encoder_weights, classes=num_classes)
    model = model.to(device)

    # criterion = nn.CrossEntropyLoss(weight= torch.tensor([1.0, 10.0], device=device))
    criterion = FocalLoss(mode='multiclass') #nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)


    # train block
    best_acc = 0.0
    num_epochs = 50

    dataloaders = {'train':train_loader, 'valid': val_loader}
    dataset_sizes = {'train':len(train_dataset), 'valid': len(val_dataset)}


    # Initialize TensorBoard SummaryWriter
    # writer_train = SummaryWriter(log_dir=os.path.join(data_dir, 'log', 'train'))
    # writer_val = SummaryWriter(log_dir=os.path.join(data_dir, 'log', 'valid'))

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
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

                    # revert padding
                    pad_size = (224 - 128) // 2
                    outputs = outputs[:, :, pad_size:pad_size+128, pad_size:pad_size+128]
                    labels = labels[:, pad_size:pad_size+128, pad_size:pad_size+128]
                    # print('after reverting padding ====', outputs.size(), labels.size())

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                exp_lr_scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / (dataset_sizes[phase]* 128 * 128)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')


            # write to tensorboard
            # writer_train.add_scalar('Loss', train_metrics['train_loss'], epoch)
            # writer_train.add_scalar('R2', train_metrics['train_R2'], epoch)

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save({'state_dict': model.state_dict()}, os.path.join(out_dir, 'model.pth.tar'))

    # save predictions on validation
    model.load_state_dict(torch.load(os.path.join(out_dir, 'model.pth.tar'))['state_dict'])
    save_predictions(model, val_loader, out_dir, device='cuda')

    print()
    print(f'Best val Acc: {best_acc:4f}')

    # writer_train.close()
    # writer_val.close()


# Example usage
if __name__ == "__main__":
    data_dir = "/app/dev/FM4EO/data/patch_data/2016"
    split_path = "/app/dev/FM4EO/data/patch_data/2016/MASK/train_test_splits.csv"
    feature_idx = [2,1,0]
    classification = 'combined'
    backbone = 'resnet34' #resnet50, resnet18, resnext50_32x4d, efficientnet-b4, efficientnet-b5, efficientnet-b6
    encoder_weights = 'imagenet'
    num_classes = 3
    out_dir = '/app/dev/FM4EO/results/unet_combined_focal'

    train_model(data_dir, split_path, feature_idx, classification, backbone, encoder_weights, num_classes, out_dir)



# # first compute statistics for true positives, false positives, false negative and
# # true negative "pixels"
# tp, fp, fn, tn = smp.metrics.get_stats(output, target, mode='multiclass', threshold=0.5)

# # then compute metrics with required reduction (see metric docs)
# iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
# f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
# f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
# accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
# recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
# precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro-imagewise")