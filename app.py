import base64
from dataclasses import fields
import os
from flask import Flask, redirect, url_for, request, jsonify
from flask_marshmallow import Marshmallow
from sklearn.preprocessing import MultiLabelBinarizer
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, String
from sqlalchemy_utils.functions import database_exists
import numpy as np
import pandas as pd
import cv2
import torch
import timm
import torch.nn as nn
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + \
    os.path.join(basedir, 'satellite.db')
MODEL_PATH = 'resnet_model.pth'
db = SQLAlchemy(app)
ma = Marshmallow(app)

# create database
def db_create():
    db.create_all()
    print('Database Created!')

#insert data
def db_seed(img_encode,prediction):
    satellite = Satellite(img=img_encode,predicted_values=prediction)
    db.session.add(satellite)
    db.session.commit()
    print('Database seeded!')

class Satellite(db.Model):
    _tablename = 'satellite'
    id = Column(Integer, primary_key=True)
    img = Column(String)
    predicted_values = Column(String)


class SatelliteSchema(ma.Schema):
    class Meta:
        fields = ('id', 'img', 'predicted_values')


satellite_schema = SatelliteSchema(many=True)

CFG = {
    "img_size": 224,
    "model_name": "resnet18d",
    "num_classes": 17,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

transform = A.Compose([
    A.Resize(CFG["img_size"], CFG["img_size"]),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(
        0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
    ToTensorV2()
],
    p=1.0)

df = pd.read_csv('train_classes.csv')
df["list_tags"] = df["tags"].str.split(" ")
encoder = MultiLabelBinarizer()
tags_train = encoder.fit_transform(df["list_tags"].values)


class ResNet18(nn.Module):

    def __init__(self, model_name, num_classes):

        super(ResNet18, self).__init__()
        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=num_classes)
        num_in_features = self.model.get_classifier().in_features
        self.model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.model.fc = nn.Sequential(nn.Flatten(),
                                      nn.Linear(num_in_features, 128),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout(p=0.2),
                                      nn.Linear(128, num_classes),
                                      nn.Sigmoid()
                                      )

    def forward(self, x):
        return self.model(x)


model = ResNet18(CFG["model_name"], CFG["num_classes"])
model.to(CFG["device"])
print(model.parameters)

# model = models.resnet18(pretrained=True)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))

thresh = [0.21, 0.07, 0.11, 0.19, 0.05, 0.16, 0.08, 0.03,
          0.13, 0.19, 0.17, 0.11, 0.27, 0.19, 0.18, 0.1, 0.18]


def pred_single(img_encode, return_label=True):
    with torch.no_grad():
        model.eval()
        # img = np.array(Image.open(img_path))
        im_bytes = base64.b64decode(img_encode)
        im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
        img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(image=img)['image']
        bs_img = img.view(1, 3, 224, 224)
        #bs_img = bs_img.to(device)
        preds = model(bs_img).float().numpy()
        prediction = (preds > thresh).astype(float)
        yhat_test_inv = encoder.inverse_transform(np.array(prediction))
        # tags = " ".join(yhat_test_inv)
        return yhat_test_inv


@app.route('/', methods=['GET'])
def index():
    return jsonify("ML")

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Make prediction
        img_encode=base64.b64encode(f.read())
        preds = pred_single(img_encode, model)
        s = ""
        for x in preds:
            s = s+str(x)+","
        s = s[:-2]
        s = s[1:]
        if database_exists(app.config["SQLALCHEMY_DATABASE_URI"]):
            db_seed(str(img_encode),s)
        else :
            db_create()
            db_seed(str(img_encode),s)
        list = s.split(',')
        return jsonify(prediction=list)
    return None

@app.route('/get_list', methods=['GET'])
def get_list():
    # get all the data from database
    if database_exists(app.config["SQLALCHEMY_DATABASE_URI"]):
        satellite_list = Satellite.query.all()
        result = satellite_schema.dump(satellite_list)
        return jsonify(result=result),200
    else :
        return jsonify(result="No Predicted Images found!"),404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
