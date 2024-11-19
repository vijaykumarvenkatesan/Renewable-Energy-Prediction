from flask import Flask, render_template, request
import pandas as pd
from model.sarima_model import train_and_forecast
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/forecast', methods=['POST'])
def forecast():
    # Retrieve form inputs for forecast settings
    power_type = request.form.get('power_type')
    forecast_unit = request.form.get('forecast_unit')
    forecast_length = request.form.get('forecast_length')  # Retrieve as a string for debugging
    
    # Debug print statements
    print("Power Type:", power_type)
    print("Forecast Unit:", forecast_unit)
    print("Forecast Length:", forecast_length)
    
    if forecast_length is None:
        return "Error: Forecast length is missing.", 400

    # Convert to integer after confirming it's not None
    forecast_length = int(forecast_length)
    
    # Continue with the rest of the logic...
    dataset_path = 'data1/solar_data.xlsx' if power_type == 'solar' else 'data1/wind_data.xlsx'
    forecast, mae, mse, rmse = train_and_forecast(dataset_path, forecast_length, forecast_unit)
    
    return render_template('forecast.html', forecast=forecast, power_type=power_type,
                           forecast_length=forecast_length, forecast_unit=forecast_unit,
                           mae=mae, mse=mse, rmse=rmse)


if __name__ == '__main__':
    app.run(debug=True)
