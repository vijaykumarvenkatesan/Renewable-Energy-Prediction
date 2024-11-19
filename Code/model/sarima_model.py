import pandas as pd 
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

import pandas as pd

def load_data(filepath):
    # Load the dataset
    data = pd.read_excel(filepath)
    
    # Convert the 'Time(year-month-day h:m:s)' column to datetime format
    data['Time(year-month-day h:m:s)'] = pd.to_datetime(data['Time(year-month-day h:m:s)'])
    
    # Set the 'Time(year-month-day h:m:s)' column as the index
    data.set_index('Time(year-month-day h:m:s)', inplace=True)
    
    # Set the frequency as hourly ('H') to ensure consistent intervals in the time series
    data = data.asfreq('H')  # Replace 'H' with your actual data frequency if different
   
    # Return the 'Power (MW)' column for forecasting
    return data['Power (MW)']


def train_and_forecast(filepath, forecast_length, forecast_unit):
    # Load dataset
    data = load_data(filepath)

    # Set SARIMA order (you can further optimize this)
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 24 if forecast_unit == 'hours' else 7)

    # Fit the SARIMA model
    model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)

    # Forecast for the specified steps
    steps = forecast_length if forecast_unit == 'hours' else forecast_length * 24
    forecast = model_fit.forecast(steps=steps)

    # Calculate model metrics
    predictions = model_fit.predict(start=0, end=len(data)-1)
    mae = mean_absolute_error(data, predictions)
    mse = mean_squared_error(data, predictions)
    rmse = np.sqrt(mse)

    return forecast, mae, mse, rmse
