import config
import dataset
import preprocessing
import model

import time
from collections import defaultdict

import numpy as np
import dill as pickle

import torch

from tqdm.auto import tqdm

from sklearn.metrics import fbeta_score

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# training class
class TrainModel(object):
    
    def __init__(self, model, optimizer, lr_scheduler):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.num_epochs = config.CFG["num_epochs"]
        
    def cuda_check(self, variable):
        print(variable.is_cuda)
        return
        
    def save_checkpoint(self, model, epoch):
        print(f'Saving checkpoint...')
        PATH = f'./resnet18_epoch_{epoch+1}.pth'
        torch.save(model.state_dict(), PATH)
        return
    
    def load_checkpoint(self, checkpoint):
        print(f'Loading checkpoint...')
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return
    
    def train_one_epoch(self, data, epoch):
        
        self.model.train()
        
        dataset_size = 0
        running_loss = 0.0
        
        bar = tqdm(enumerate(data), total = len(data), position = 0, leave = True)
        for step, (img, label) in bar:
            
            img = img.to(config.CFG["device"])
            label = label.to(config.CFG["device"])
            
            ypred = self.model(img)
            loss = model.criterion(ypred, label)
            
            if config.CFG["accumulation_steps"] > 1:
                loss = loss / config.CFG["accumulation_steps"]
                
            loss.backward()
            
            if (step+1) % config.CFG["accumulation_steps"] == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            batch_size = img.size(0)
            dataset_size += batch_size
            running_loss += (loss.item() * batch_size)
            epoch_loss = running_loss / dataset_size
            
            ypred_thresh = (ypred.detach().cpu().numpy() > 0.2).astype(float)
            score = fbeta_score(label.detach().float().cpu().numpy(), ypred_thresh, beta = 2, average = "samples")
            
            bar.set_description(f"Training epoch [{epoch+1}/{self.num_epochs}]")
            bar.set_postfix(Epoch = epoch+1, Training_loss = epoch_loss, Training_score = score, LR = self.optimizer.param_groups[0]["lr"])
        
        del dataset_size, running_loss, batch_size
        torch.cuda.empty_cache()
        
        return epoch_loss, score
    
    def val_one_epoch(self, data, epoch):
        
        self.model.eval()
        
        dataset_size = 0
        running_loss = 0.0
        
        y_val_batch = []
        yhat_val_batch = []
        
        bar = tqdm(enumerate(data), total = len(data), position = 0, leave = True)
        for step, (img, label) in bar:
            
            with torch.no_grad():
                
                img = img.to(config.CFG["device"])
                label = label.to(config.CFG["device"])

                ypred = self.model(img)
                loss = model.criterion(ypred, label)
                
            batch_size = img.size(0)
            dataset_size += batch_size
            running_loss += (loss.item() * batch_size)
            epoch_loss = running_loss / dataset_size
            
            ypred_thresh = (ypred.detach().cpu().numpy() > 0.2).astype(float)
            score = fbeta_score(label.detach().float().cpu().numpy(), ypred_thresh, beta = 2, average = "samples")
            
            y_val_batch.extend(label.detach().float().cpu().numpy())
            yhat_val_batch.extend(ypred.detach().float().cpu().numpy())
            
            bar.set_description(f"Validation epoch [{epoch+1}/{self.num_epochs}]")
            bar.set_postfix(Epoch = epoch+1, Validation_loss = epoch_loss, Validation_score = score, LR = self.optimizer.param_groups[0]["lr"])
        
        del dataset_size, running_loss, batch_size
        torch.cuda.empty_cache()
        
        return epoch_loss, score, y_val_batch, yhat_val_batch
    
    def train(self):
        
        if torch.cuda.is_available():
            print(f'GPU Info : {torch.cuda.get_device_name(0)}')
            
        train_df = dataset.PlanetAmazonDataset(preprocessing.df_train, preprocessing.TRAIN_IMAGE_DIR, preprocessing.tags_train, transform = config.train_transform)
        train_data = dataset.train_loader(train_df)
        
        val_df = dataset.PlanetAmazonDataset(preprocessing.df_val, preprocessing.TRAIN_IMAGE_DIR, preprocessing.tags_val, transform = config.val_transform)
        val_data = dataset.val_loader(val_df)
            
        training_log = defaultdict(list)
        best_epoch_loss = np.inf
        best_epoch_score = 0.0
        
        start = time.time()
        
        for epoch in range(config.CFG["num_epochs"]):
            
            train_loss, train_score = self.train_one_epoch(train_data, epoch)
            val_loss, val_score, y_val_epoch, yhat_val_epoch = self.val_one_epoch(val_data, epoch)
            
            training_log["train_loss"].append(train_loss)
            training_log["train_score"].append(train_score)
            training_log["val_loss"].append(val_loss)
            training_log["val_score"].append(val_score)
            training_log["y_val_epoch"] = np.array(y_val_epoch)
            training_log["yhat_val_epoch"] = np.array(yhat_val_epoch)
            
            self.lr_scheduler.step()
            
            if val_loss < best_epoch_loss:
                print(f'Validation loss reduced : {best_epoch_loss} ------> {val_loss}')
                best_epoch_loss = val_loss
                
            if val_score > best_epoch_score:
                print(f"Validation score improved : {best_epoch_score} ------> {val_score}")
                best_epoch_score = val_score
                
                checkpoint = {
                    "model_state_dict" : self.model.state_dict(),
                    "optimizer_state_dict" : self.optimizer.state_dict()
                }
                
                self.save_checkpoint(self.model, epoch)
                
        training_log["best_val_loss"] = best_epoch_loss
        training_log["best_val_score"] = best_epoch_score
        end = time.time()
        lapse = end - start
        print(f'Training completed in {lapse // 3600} hours {(lapse % 3600) // 60} minutes {(lapse % 3600) % 60} seconds')
        print(f'Best validation score : {best_epoch_score}')
        
        self.load_checkpoint(checkpoint)
        pickle.dump(training_log, open('./training_log.pkl', 'wb'))
        
        return self.model, self.optimizer, self.lr_scheduler, training_log


resnet_model = TrainModel(model, model.optimizer, model.lr_scheduler)
resnet_model_trained, optimizer_trained, lr_scheduler_trained, training_log = resnet_model.train()


# plot training loss, score and validation loss, score vs epoch
def plot_training_log(training_log):
    
    fig = make_subplots(rows = 1, cols = 2, subplot_titles = ("Loss", "Fbeta scores"))
    
    fig.add_trace(
        go.Scatter(
            x = list(range(1, config.CFG["num_epochs"] + 1)), 
            y = training_log["train_loss"], 
            name = "Training_loss"), 
        row = 1, col = 1)
    
    fig.add_trace(
        go.Scatter(
            x = list(range(1, config.CFG["num_epochs"] + 1)), 
            y = training_log["train_score"], 
            name = "Training_score"), 
        row = 1, col = 2)
    
    fig.add_trace(
        go.Scatter(
            x = list(range(1, config.CFG["num_epochs"] + 1)), 
            y = training_log["val_loss"], 
            name = "Validation_loss"), 
        row = 1, col = 1)
    
    fig.add_trace(
        go.Scatter(
            x = list(range(1, config.CFG["num_epochs"] + 1)), 
            y = training_log["val_score"], 
            name = "Validation_score"), 
        row = 1, col = 2)
    
    fig.show()
    
training_log = pickle.load(open('./training_log.pkl', 'rb'))
plot_training_log(training_log)