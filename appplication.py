from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            airline=request.form.get('airline'),
            source_city=request.form.get('source_city'),
            departure_time=request.form.get('departure_time'),
            stops=request.form.get('stops'),
            arrival_time=request.form.get('arrival_time'),
            destination_city=request.form.get('destination_city'),
            class_of_travel=request.form.get('class'),
            duration=float(request.form.get('duration')),
            days_left=float(request.form.get('days_left')))
        
        pred_df=data.get_data_as_data_frame()

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=int(results[0]))
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        

