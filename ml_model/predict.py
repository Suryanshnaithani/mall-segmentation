#predict using trained model   
import pandas as pd
import joblib

def predict(input_data):
    kmeans = joblib.load('ml_model/kmeans_model.pkl')
    return kmeans.predict(input_data)
