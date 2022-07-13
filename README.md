# __Satellite Image Terrain Predictor Server__
## Flask API to predict the terrain type of the satellite images.

* Language : `Python`
* Database : `SQLAlchemy` (*satellite_predictor.db*)

## API Documentation :

### __Authentication__

#### (Using Flask Login)

> POST `/signup`
>> REQUEST :
>>> name : String

>>> email : String

>>> password : String

>> RESPONSE :
>>> JSON Message indicating whether User is registered or not.

> POST `/login`
>> REQUEST :
>>> email : String

>>> password : String

>>> remember : Boolean

>> RESPONSE :
>>> JSON Message indicating whether User is logged in or not.

> GET `/logout`
>>__@login_required__

>> RESPONSE :
>>> JSON Message indicating whether User is logged out or not.

### Satellite List

> GET `/get_list`
>>__@login_required__
>> Gives the list of all the images which have been predicted by the current User (History)

> POST `/predict`
>>__@login_required__
>> Takes the satellite image as request and gives the predicted results as response.


## Database Tables

```SQL
__tablename__ = 'satellite'
satellite_id = Column(Integer, primary_key=True)
img = Column(String)
predicted_values = Column(String)
user_id = Column(Integer)
```

```SQL
 __tablename__ = 'users'
id = Column(Integer, primary_key=True) 
name = Column(String)
email = Column(String, unique=True)
password = Column(String)
```

## Data Training and Prediction Model

* [Training Dataset Link](https://www.kaggle.com/datasets/nikitarom/planets-dataset/code?select=planet)

* [Dataset Model](/resnet_model.pth)

## Libraries Used
```r
albumentations==1.1.0
Flask==2.1.2
flask-marshmallow==0.14.0
Flask-SQLAlchemy==2.5.1
Flask-Login==0.6.1
numpy==1.23.0
opencv-python-headless==4.6.0.66
pandas==1.4.3
sklearn==0.0
SQLAlchemy==1.4.39
SQLAlchemy-Utils==0.38.2
timm==0.5.4
torch==1.11.0
torchvision==0.12.0
gunicorn==20.1.0
```