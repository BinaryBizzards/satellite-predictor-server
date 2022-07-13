# __Satellite Image Terrain Predictor Server__
## Flask API to predict the terrain type of the satellite images.

* Language : `Python`
* Database : `SQLAlchemy` (*satellite_predictor.db*)

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


## API :

### __Authentication__

#### (Using Flask Login)

> POST `/signup` <br /> 
> Register User

> POST `/login` <br /> 
> User Login

> GET `/logout` <br /> 
>__@login_required__  <br />  Sign Out User

### Satellite List

> GET `/get_list` <br /> 
>__@login_required__ <br /> Gives the list of all the images which have been predicted by the current User (History)

> POST `/predict` <br /> 
>__@login_required__ <br /> Takes the satellite image as request and gives the predicted results as response.
