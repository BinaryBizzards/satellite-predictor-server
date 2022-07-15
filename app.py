import base64
from dataclasses import fields
import os
import re
from flask import Flask, request, jsonify
from flask_marshmallow import Marshmallow
from flask_login import login_user, login_required, logout_user, current_user
from flask_login import UserMixin, LoginManager
# from flask_jwt_extended import JWTManager, jwt_required, create_access_token
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
from werkzeug.security import generate_password_hash, check_password_hash


app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + \
    os.path.join(basedir, 'satellite_predictor.db')

app.config['SECRET_KEY'] = 'Satelitain@6877'

MODEL_PATH = 'resnet_model.pth'
db = SQLAlchemy(app)
ma = Marshmallow(app)
lm = LoginManager(app)

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
    __tablename__ = 'satellite'
    satellite_id = Column(Integer, primary_key=True)
    img = Column(String)
    predicted_values = Column(String)


class SatelliteSchema(ma.Schema):
    class Meta:
        fields = ('satellite_id', 'img', 'predicted_values')

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True) 
    name = Column(String)
    email = Column(String, unique=True)
    password = Column(String)

class UserSchema(ma.Schema) :
    class Meta:
        fields=('id' , 'name' , 'email' , 'password')

user_schema = UserSchema()
users_schema = UserSchema(many=True)
satellite_schema = SatelliteSchema(many=True)

@lm.user_loader
def load_user(id) :
    return User.query.get(int(id))


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

def validate(email):
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    if(re.fullmatch(regex, email)):
        return True
    return False

@app.route('/', methods=['GET'])
def index():    
    return jsonify("Satellitain")

@app.route('/login', methods=['POST'])
def login():
    if not database_exists(app.config["SQLALCHEMY_DATABASE_URI"]):
        db_create()
    email = request.form['email']
    password = request.form['password']
    remember = True if request.form['remember'] else False
    if not email or not password :
        return jsonify("Please fill all the fields"), 400
    user = User.query.filter_by(email=email).first()
    if not user :
        return jsonify("User is not registered yet"), 404
    if not check_password_hash(user.password, password) :
        return jsonify("Incorrect Password"), 400
    login_user(user, remember=remember)
    return jsonify("User Logged in Successfully")
    

@app.route('/signup', methods=['POST'])
def signup():
    if not database_exists(app.config["SQLALCHEMY_DATABASE_URI"]):
        db_create()
    email = request.form['email']
    name = request.form['name']
    password = request.form['password']
    if not email or not password or not name:
        return jsonify("Please fill all the fields"), 400
    if not validate(email) :
        return jsonify("Enter a Valid Email ID"), 400
    if len(password)<6 :
        return jsonify("Password must have minimum 6 characters"), 400
    if len(name)<3 :
        return jsonify("Enter a Valid Name"), 400 
    user = User.query.filter_by(email=email).first()
    if (user) :
        return jsonify("User already registered"), 409
    new_user = User(email=email, name=name, password=generate_password_hash(password, method='sha256'))
    db.session.add(new_user)
    db.session.commit()
    return jsonify("User registered Successfully")

@app.route('/logout', methods=['GET','POST'])
@login_required
def logout():
    logout_user()
    return jsonify("Account signed out")

@app.route('/profile', methods=['GET'])
@login_required
def profile() :
    return jsonify(name=current_user.name, email=current_user.email)

@app.route('/predict', methods=['GET', 'POST'])
# @login_required
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
        img_encode=img_encode[:-1]
        img_encode=img_encode[2:]
        if database_exists(app.config["SQLALCHEMY_DATABASE_URI"]):
            db_seed(str(img_encode),s)
        else :
            db_create()
            db_seed(str(img_encode),s)
        list = s.split(',')
        return jsonify(prediction=list)
    return None

@app.route('/get_list', methods=['GET'])
# @login_required
def get_list():
    # get all the data from database
    if database_exists(app.config["SQLALCHEMY_DATABASE_URI"]):
        satellite_list = Satellite.query.all()
        result = satellite_schema.dump(satellite_list)
        return jsonify(result),200
    else :
        return jsonify("No Predicted Images found!"),404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')