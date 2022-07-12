import config

import os

import numpy as np

import cv2
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

# error handling class
class PlanetAmazonDatasetError(Exception):
    pass

# dataset class
class PlanetAmazonDataset(Dataset):
    
    def __init__(self, csv_file, img_path, tags, transform = None, is_train = True, idx_tta = None):
        self.csv_file = csv_file
        
        if isinstance(img_path, str):
            self.img_paths = [img_path]
            
        elif isinstance(img_path, (list, tuple)):
            self.img_paths = img_path
            
        else:
            raise PlanetAmazonDatasetError(f"Path type must be str, list or tuple, got {type(img_path)}")
            
        self.tags = tags
        self.transform = transform
        self.is_train = is_train
        self.idx_tta = idx_tta
        self.flag = 0
        
    def __getitem__(self, idx):
        
        img_name = self.csv_file.iloc[idx]["image_name"] + ".jpg"
        
        for path in self.img_paths:
            if img_name in os.listdir(path):
                img_file_path = os.path.join(path, img_name)
                self.flag = 1
                break
                
            if self.flag == 0:
                continue
            
        if self.flag == 0:
            raise PlanetAmazonDatasetError(f"Cannot fetch {img_name} in {self.img_paths}")
            
        self.flag = 0
                
        img = cv2.cvtColor(cv2.imread(img_file_path), cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.custom_augment(img)
            img = self.transform(img)
            
        label = torch.tensor(self.tags[idx])
        return (img.float(), label.float())
    
    def __len__(self):
        return len(self.csv_file)
    
    def collate_fn(self, batch):
        
        imgs, labels = [], []
        
        for (img, label) in batch:
            img = torch.tensor(img)
            if self.transform:
                img = self.transform(image = img)["image"]
            imgs.append(img[None])
#                 print(img.shape)
            labels.append(label)
            
        imgs = torch.cat(imgs).float().to(config.CFG["device"])
#         imgs = torch.tensor(np.array(imgs)).float().to(CFG["device"])
        labels = torch.tensor(np.array(labels)).float().to(config.CFG["device"])
        return (imgs, labels)
        
    # function for tta
    def custom_augment(self, img):
        
        choice = np.random.randint(0, 6) if self.is_train else self.idx_tta
        
        if choice == 0:
            # rotate 90
            img = cv2.rotate(img, rotateCode = cv2.ROTATE_90_CLOCKWISE)
            
        if choice == 1:
            # rotate 90 and horizontalflip
            img = cv2.rotate(img, rotateCode = cv2.ROTATE_90_CLOCKWISE)
            img = cv2.flip(img, flipCode = 1)
            
        if choice == 2:
            # rotate 180
            img = cv2.rotate(img, rotateCode = cv2.ROTATE_180)
            
        if choice == 3:
            # rotate 180 and horizontalflip
            img = cv2.rotate(img, rotateCode = cv2.ROTATE_180)
            img = cv2.flip(img, flipCode = 1)
            
        if choice == 4:
            # rotate 90 counter-clockwise
            img = cv2.rotate(img, rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        if choice == 5:
            # rotate 90 counter-clockwise and horizontalflip
            img = cv2.rotate(img, rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE)
            img = cv2.flip(img, flipCode = 1)
        
        return img
    
    def plot_image(self, idx, ax = None):
        img, label = self[idx]
        tag = self.csv_file.iloc[idx]["tags"]
        title = f"{tag} - {label}"
        
        if ax is None:
            plt.imshow(img)
            plt.title(title)
            
        else:
            ax.imshow(img)
            ax.set_title(title)


# dataloader class
class Dataloader(object):
    
    def __init__(self, mode):
        self.mode = mode
        
    def __call__(self, df):
        
        if self.mode == "train":
            loader = DataLoader(dataset = df, 
                                batch_size = config.CFG["train_batch_size"], 
                                shuffle = True, 
                                num_workers = 2, 
                                pin_memory = True, 
#                                 collate_fn = dataset.collate_fn
                               )
            return loader
        
        elif self.mode == "val":
            loader = DataLoader(dataset = df, 
                                batch_size = config.CFG["val_batch_size"], 
                                shuffle = False, 
                                num_workers = 2, 
                                pin_memory = True, 
#                                 collate_fn = dataset.collate_fn
                               )
            return loader
        
train_loader = Dataloader("train")
val_loader = Dataloader("val")

# plot one transformed image
sample_dataset = PlanetAmazonDataset(config.df_train, config.TRAIN_IMAGE_DIR, config.tags_train, config.train_transform)
img, label = next(iter(train_loader(sample_dataset)))
print(img[0].shape)
plt.imshow(img[0].permute(1, 2, 0).cpu().numpy())
plt.title(f'{label[0]}')

# check tensor shape for sanity
print(img.shape, label.shape)