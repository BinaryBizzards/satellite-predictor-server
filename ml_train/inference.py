import config
import dataset
import preprocessing
import train
import model_improvement

import os

import numpy as np
import pandas as pd

from tqdm.auto import tqdm

import torch


# load the test dataset
TEST_DIR_1 = "../input/planets-dataset/planet/planet/test-jpg"
TEST_DIR_2 = "../input/planets-dataset/test-jpg-additional/test-jpg-additional"
test_length = len(os.listdir(TEST_DIR_1)) + len(os.listdir(TEST_DIR_2))
df_test = pd.read_csv("../input/planets-dataset/planet/planet/sample_submission.csv")
assert df_test.shape[0] == test_length, "file lengths not matching"

print(len(df_test))

tags_test = np.zeros((test_length, config.CFG["num_classes"]))
test_transform = config.Transform("val")
test_loader = dataset.Dataloader("val")


# prediction function for test dataset
def predict_test():
    
    train.resnet_model_trained.eval()
    yhat = []
    
    for idx_tta in range(6):
        
        test_df = dataset.PlanetAmazonDataset(df_test, [TEST_DIR_1, TEST_DIR_2], tags_test, transform = test_transform, is_train = False, idx_tta = idx_tta)
        test_data = test_loader(test_df)
        
        yhat_tta = []
    
        bar = tqdm(enumerate(test_data), total = len(test_data), position = 0, leave = True)
    
        for step, (img, _) in bar:
            img = img.to(config.CFG["device"])
            with torch.no_grad():
                ypred = train.resnet_model_trained(img)
    #         print(ypred_label.shape)
            yhat_tta.extend(ypred.detach().float().cpu().numpy())
#         print(len(yhat))

        yhat.append(yhat_tta)
        
#         assert len(yhat) == len(df_test), "length of predicted test data not matching"
    yhat_test = np.mean(np.array(yhat), axis = 0)
    yhat_test_label = (yhat_test > model_improvement.threshs).astype(float)
    return yhat_test_label

test_pred = predict_test()


# transform the predicted probabilities to classes and create the submission file
yhat_test_inv = preprocessing.encoder.inverse_transform(test_pred)
test_tags = []
for row in yhat_test_inv:
    tags = " ".join(row)
    test_tags.append(tags)
    
df_test["tags"] = test_tags
df_test.to_csv("my_sample_submission.csv", index = False)