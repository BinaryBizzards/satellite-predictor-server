FROM python:3.10
WORKDIR /predict_img

ADD . /predict_img

RUN pip install -r requirements.txt

CMD ["python","app.py"]