import preprocessing

import os
from collections import Counter

import pandas as pd

import cv2
import plotly.express as px
import matplotlib.pyplot as plt

# separate classes from tags and plot them separately
row_tags = preprocessing.df["list_tags"].values
tags = [tag for row in row_tags for tag in row]
counter_tags = Counter(tags)

df_tags = pd.DataFrame({
    "tag" : counter_tags.keys(),
    "total" : counter_tags.values()
}).sort_values("total")

fig = px.bar(df_tags, x = "total", y = "tag", color = "total", orientation = "h")
fig.update_layout(title = "Class distribution")
fig.show()

# plot one image from each class
all_tags = list(set(tags))
N_tags = len(all_tags)
fig, axes = plt.subplots(4, (N_tags//4)+1, figsize=(20, 20))
for idx, tag in enumerate(all_tags):
    filename = preprocessing.loc[preprocessing.df["tags"].str.contains(tag)].image_name.values[0]
    img = cv2.imread(os.path.join(preprocessing.TRAIN_IMAGE_DIR, filename+".jpg"))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    idx_col = idx // 4
    idx_row = idx % 4
    axes[idx_row][idx_col].set_title(tag)
    axes[idx_row][idx_col].imshow(img)
axes[1][-1].remove()
axes[2][-1].remove()
axes[3][-1].remove()

# check the percentage distribution of rare classes in train and validation data
rare_classes = ["bare_ground", "artisinal_mine", "blooming", "conventional_mine", "slash_burn", "blow_down", "selective_logging"]

for rare in rare_classes:
    
    total_train = preprocessing.df_train.loc[preprocessing.df_train["tags"].str.contains(rare)].shape[0]
    total_val = preprocessing.df_val.loc[preprocessing.df_val["tags"].str.contains(rare)].shape[0]
    
    print(f'Train {rare} : {100 * total_train / preprocessing.df_train.shape[0]:.4f}% ({total_train})')
    print(f'Val {rare} : {100 * total_val / preprocessing.df_train.shape[0]:.4f}% ({total_val})\n')