import config

import torch
import torch.nn as nn

import timm

from torchsummary import summary

# model class
class ResNet18(nn.Module):
    
    def __init__(self, model_name, num_classes):
        
        super(ResNet18, self).__init__()
        self.model = timm.create_model(model_name, pretrained = True, num_classes = num_classes)
        num_in_features = self.model.get_classifier().in_features
        self.model.avgpool = nn.AdaptiveAvgPool2d(output_size = (1, 1))
        self.model.fc = nn.Sequential(nn.Flatten(), 
                                      nn.Linear(num_in_features, 128), 
                                      nn.ReLU(inplace = True), 
                                      nn.Dropout(p = 0.2), 
                                      nn.Linear(128, num_classes), 
                                      nn.Sigmoid()
                                     )
        
    def forward(self, x):
        return self.model(x)
    
model = ResNet18(config.CFG["model_name"], config.CFG["num_classes"])
model.to(config.CFG["device"])
summary(model, (3, config.CFG["img_size"], config.CFG["img_size"]))


# define optimizer, loss_function and lr_scheduler
optimizer = torch.optim.Adam(model.parameters(), lr = config.CFG["learning_rate"])
criterion = nn.BCELoss()
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = config.CFG["num_epochs"], eta_min = 0)