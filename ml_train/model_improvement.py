import preprocessing
import train

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tqdm.auto import tqdm

from sklearn.metrics import fbeta_score, confusion_matrix


# check if the threshold set is able to filter positive and negative classes in the validation set properly
y_val, yhat_val = train.training_log["y_val_epoch"], train.training_log["yhat_val_epoch"]
pos_probas, neg_probas = [], []

for class_, idx in preprocessing.encoder._cached_dict.items():
    pos_probas.append(yhat_val[np.where(y_val[:, idx] != 0), idx].mean())
    neg_probas.append(yhat_val[np.where(y_val[:, idx] == 0), idx].mean())
    
go.Figure([
    go.Bar(x = list(preprocessing.encoder._cached_dict), y = pos_probas, name = "yhat probas | y = 1"),
    go.Bar(x = list(preprocessing.encoder._cached_dict), y = neg_probas, name = "yhat probas | y = 0")
]).show()


# find the best threshold for each class
def find_best_threshold(yhat, y):
    n_tags = y.shape[1]
    best_threshs = [0.2] * n_tags
    resolution = 100
    for jdx in tqdm(range(n_tags)):
        best_score = train.training_log["best_val_score"]
        threshs = best_threshs.copy()
        for kdx in range(resolution):
            kdx /= resolution
            threshs[jdx] = kdx
            yhat_thresh = (yhat > threshs).astype(float)
            score = fbeta_score(y, yhat_thresh, beta = 2, average = "samples")
            if score > best_score:
                best_score = score
                best_threshs[jdx] = kdx
                
    global_best_score = fbeta_score(y, yhat_thresh, beta = 2, average = "samples")
    print(f"Threshs : {best_threshs} ---- Best score : {global_best_score}")
    
    return best_threshs
    
threshs = find_best_threshold(yhat_val, y_val)


# plot fbeta_score of each class
class_scores = {}
classes = preprocessing.encoder.classes_

for jdx in range(y_val.shape[1]):
    y = y_val[:, jdx].ravel()
    yhat = (yhat_val[:, jdx].ravel() > threshs[jdx]).astype(float)
    score = fbeta_score(y, yhat, beta = 2)
    class_scores[classes[jdx]] = round(score, 4)
    
df_score = pd.DataFrame(dict(label = list(class_scores.keys()), score = list(class_scores.values()))).sort_values("score", ascending = False)

fig = px.bar(df_score, x = "label", y = "score", color = "score")
fig.show()


# plot confusion_matrix of each class
fig = make_subplots(cols=5, rows=4)
for jdx in range(y_val.shape[1]):
    y = y_val[:, jdx].ravel()
    yhat = (yhat_val[:, jdx].ravel() > threshs[jdx]).astype(float)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    mat = np.array([[fn, tn], [tp, fp]])
    col = jdx // 4+1
    row = jdx % 4+1
    fig.add_trace(
        go.Heatmap(
            z=mat, text=[[f"fn: {fn}", f"tn: {tn}"], [f"tp: {tp}", f"fp: {fp}"]], 
            texttemplate="%{text}", colorscale='Viridis', name = preprocessing.encoder.classes_[jdx],
            showscale=False
        ),
        col=col, row=row, 
    )
    fig.update_xaxes(title = preprocessing.encoder.classes_[jdx], showticklabels=False, row=row, col=col)
    fig.update_yaxes(showticklabels=False, row=row, col=col)
    

fig.update_layout(
    width=1200, height=800, title="Confusion matrices", 
)
fig.show()