# satellite-predictor-server
Flask API to predict the terrain type of the satellite images.

Language: Python
Database: SQLAlchemy

GET /get_list
Gives the list of all the images which have been predicted

POST/predict
Takes the satellite image as request and gives the predicted results as response.
