import config

import os

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# set seeds
config.set_seed(config.CFG["seed"])

# read the dataset
ROOT_DIR = "../input/planets-dataset/planet/planet"
TRAIN_IMAGE_DIR = os.path.join(ROOT_DIR, "train-jpg")
print(f'Number of training examples : {len(os.listdir(TRAIN_IMAGE_DIR))}')
TRAIN_CSV = os.path.join(ROOT_DIR, "train_classes.csv")

df = pd.read_csv(TRAIN_CSV)
if config.CFG["debug"]:
    df = df.sample(n = 1000, random_state = config.CFG["seed"]).reset_index(drop = True)
df.head()

# preprocessing the dataset
df["list_tags"] = df["tags"].str.split(" ")

df_train, df_val = train_test_split(df, test_size = config.CFG["val_frac"], shuffle = True, random_state = config.CFG["seed"])

encoder = MultiLabelBinarizer()
tags_train = encoder.fit_transform(df_train["list_tags"].values)
tags_val = encoder.transform(df_val["list_tags"].values)