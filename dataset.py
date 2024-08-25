import os, glob
import pandas as pd
from typing import Optional
import torch
# import pytorch_lightning as pl
# from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
import numpy as np
import rasterio as rio



class ImageMaskDataset(Dataset):
    def __init__(self, data_dir: str, split_path: str, mode:str,
                 feature_idx: list, classification:str, cocoa_threshold:float, 
                 transform: Optional[transforms.Compose] = None):
        
        """
        params
            data_dir: path to root folder containing image and mask
            split_path: csv file containing indices used for training
            mode: one of train, validation
            feature_idx: list of indices for feature subsetting
            classification: one of "cocoa, mining, combined"
            cocoa_threshold: probability used to categorize cocoa
                             and non-cocoa
            transform: data augmentation
        """

        self.data_dir = data_dir
        self.split_path = split_path
        self.mode = mode
        self.feature_idx = feature_idx
        self.cocoa_threshold = cocoa_threshold
        self.transform = transform
        self.classification = classification

        # norm values
        self.norm_min = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0 , 
                    -14.333027 ,-20.413631, 114.03641])
        self.norm_max = np.array([ 1014., 1626., 2104., 2408.,  2797., 3255. 
                    ,3454., 3587., 3412., 2730., -2.4528444,
                    -8.558943, 405.077])
        
        if self.feature_idx is not None:
            self.norm_min = self.norm_min[feature_idx]
            self.norm_max = self.norm_max[feature_idx]
        

        # get splits for training and testing
        self.splits_df = pd.read_csv(self.split_path)
        if self.mode is not None:
            self.file_idx = self.splits_df[self.splits_df['split'] == self.mode]
        else:
            self.file_idx = self.file_idx['patch_name']
        self.file_idx = self.file_idx['patch_name'].to_list()

        ## remove extreme edge patches. dimension 13x36
        self.file_idx  = [item for item in self.file_idx  if not 
                          any(s in item for s in ['1344', '1409', '1640', 
                                                  '1694','2132', '2133', 
                                                  '2134'])]
        
    def __len__(self):
        return len(self.file_idx)

    def __getitem__(self, idx: int) :
        image_id = self.file_idx[idx]

        # read image
        with rio.open(os.path.join(self.data_dir, 'IMAGE', image_id.replace('MASK_', 'IMG_'))) as src:
            image = src.read()
            image = np.where(image == src.nodata, 0, image)

            # normalize
            norm_image = np.zeros_like(image[self.feature_idx])
            for i in range(norm_image.shape[0]):
                norm_image[i,:,:] = (image[i,:,:]- self.norm_min[i]) / (self.norm_max[i] - self.norm_min[i])

        # read mask
        with rio.open(os.path.join(self.data_dir, 'MASK', image_id)) as src:

            # handle no data
            mask_array  =  src.read()
            mask_array = np.where(mask_array == src.nodata, 0, mask_array)
            

            if self.classification == 'cocoa':
                cocoa_mask = np.where(mask_array[1] >= self.cocoa_threshold, 1, 0) 
                mask = cocoa_mask

            elif self.classification == 'mining':
                mining_mask = np.where(mask_array[0] >1, 0, mask_array[0]) 
                mask = mining_mask
                
            elif self.classification == 'combined':
                # merge mining and cocoa mask
                mining_mask = np.where(mask_array[0] >1, 0, mask_array[0]) 
                # NB cocoa mask set to 2 for multi-class case
                cocoa_mask = np.where(mask_array[1] >= self.cocoa_threshold, 2, 0)
                mask = np.where(mining_mask !=1, cocoa_mask, mining_mask)

        if self.transform:
            norm_image = self.transform(image=np.moveaxis(norm_image,0,-1))["image"]
            norm_image = np.moveaxis(norm_image, -1, 0)
            mask = self.transform(image=mask)["image"]

        norm_image = torch.tensor(norm_image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long)

        return norm_image, mask,  image_id