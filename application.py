from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app=application
#Route for a home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            MinTemp=float(request.form.get('MinTemp')),
            MaxTemp=float(request.form.get('MaxTemp')),
            Rainfall=float(request.form.get('Rainfall')),
            Evaporation=float(request.form.get('Evaporation')),
            Sunshine=float(request.form.get('Sunshine')),
            WindGustDir=request.form.get('WindGustDir'),
            WindGustSpeed=float(request.form.get('WindGustSpeed')),
            WindDir9am=request.form.get('WindDir9am'),
            WindDir3pm=request.form.get('WindDir3pm'),
            WindSpeed9am=float(request.form.get('WindSpeed9am')),
            Humidity9am=float(request.form.get('Humidity9am')),
            Humidity3pm=float(request.form.get('Humidity3pm')),
            Cloud9am=float(request.form.get('Cloud9am')),
            Cloud3pm=float(request.form.get('Cloud3pm')),
            Temp3pm=float(request.form.get('Temp3pm')),
            RainToday=request.form.get('RainToday')
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html', results=results[0])

if __name__=="__main__":
    app.run(host='0.0.0.0')