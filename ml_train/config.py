import os

import numpy as np

import torch

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# define some variables
CFG = {
    "seed" : 42,
    "debug" : False,
    "val_frac" : 0.1,
    "img_size" : 224,
    "train_batch_size" : 512,
    "val_batch_size" : 512,
    "model_name" : "resnet18d",
    "num_classes" : 17,
    "learning_rate" : 5e-4,
    "num_epochs" : 5,
    "accumulation_steps" : 2,
    "device" : torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# set seeds for reproducibility
def set_seed(seed = 42):
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    
set_seed(CFG["seed"])

# image augmentation
class Transform(object):
    
    def __init__(self, mode):
        self.mode = mode
        
    def __call__(self, img):
        
        if self.mode == "train":
            transform = A.Compose([
                A.Resize(CFG["img_size"], CFG["img_size"]),
                A.HorizontalFlip(p = 0.5),
                A.VerticalFlip(p = 0.5),
                A.Rotate(p = 0.5),
                A.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225), max_pixel_value = 255.0, p = 1.0),
                ToTensorV2()
            ],
            p = 1.0)
            return transform(image = img)["image"]
        
        elif self.mode == "val":
            transform = A.Compose([
                A.Resize(CFG["img_size"], CFG["img_size"]),
                A.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225), max_pixel_value = 255.0, p = 1.0),
                ToTensorV2()
            ], 
            p = 1.0)
            return transform(image = img)["image"]
        
train_transform = Transform("train")
val_transform = Transform("val")