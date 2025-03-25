from flask import Blueprint, jsonify, request
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA

recorded_data = pd.DataFrame(columns=['Timestamp', 'ph', 'Turbidity', 'Conductivity', 'temperature', 'predicted_potability'])
api_bp = Blueprint('api', __name__)
model = joblib.load('utils/potability model.model')
scaler = joblib.load('utils/feature_scaler.pkl')


@api_bp.route('/status')
def api_status():
    return jsonify(status='ok')

@api_bp.route('/add_data', methods=['POST'])
def read_data():
    try:
        data = request.get_json()
        # Extract values from JSON data
        timestamp = datetime.now()
        ph = float(data.get('ph', 0))
        turbidity = float(data.get('turbidity', 0))
        conductivity = float(data.get('conductivity', 0))
        temperature = float(data.get('temperature', 0))
        predicted_potability = 0

        # Create a new row as a dictionary
        new_row = {
            'Timestamp': timestamp,
            'ph': ph,
            'Turbidity': turbidity,
            'Conductivity': conductivity,
            'temperature': temperature,
            'predicted_potability': predicted_potability
        }

        # Append the new row to the global DataFrame
        global recorded_data
        recorded_data = pd.concat([recorded_data, pd.DataFrame([new_row])], ignore_index=True)

        # Include the data in the response
        data['timestamp'] = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        return jsonify(data)
    
    except Exception as e:
        return jsonify(error=str(e))
    
    
@api_bp.route('/ml_model', methods=['POST'])
def ml_model():
    try:
        # Load the data from the global DataFrame
        global recorded_data
        data = recorded_data.copy()
        
        # Drop the timestamp column
        data = data.drop(columns=['Timestamp', 'predicted_potability', 'temperature'])
        
        #scale the data
        data = scaler.transform(data[['ph', 'Conductivity', 'Turbidity']])
        
        # Predict the potability of the water
        predicted_potability = model.predict(data)
        
        # Update the global DataFrame with the predicted potability
        recorded_data['predicted_potability'] = predicted_potability
        
        # Include the predicted potability in the response
        response = recorded_data.to_dict(orient='records')
        return jsonify(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(error=str(e))


@api_bp.route('/forecast', methods=["POST"])
def forecast():
    try:
        global recorded_data
        
        if len(recorded_data) < 10:  # Need sufficient data for forecasting
            return jsonify({"error": "Not enough data for forecasting. Need at least 10 data points."})
        
        # Number of time periods to forecast
        forecast_periods = request.json.get('periods', 5)
        
        results = {}
        features = ['ph', 'Turbidity', 'Conductivity', 'temperature', 'predicted_potability']
        
        for feature in features:
            # Convert data to time series
            series = recorded_data[feature].values
            
            # Fit ARIMA model - using simple (1,1,1) parameters
            # In production, you might want to use auto_arima to find optimal parameters
            model = ARIMA(series, order=(1, 1, 1))
            model_fit = model.fit()
            
            # Make forecast
            forecast = model_fit.forecast(steps=forecast_periods)
            
            # Store forecasted values
            results[feature] = forecast.tolist()
        
        # Add timestamp information
        last_timestamp = recorded_data['Timestamp'].iloc[-1]
        forecast_timestamps = []
        
        # Generate future timestamps based on average time difference
        if len(recorded_data) > 1:
            avg_timedelta = (recorded_data['Timestamp'].iloc[-1] - recorded_data['Timestamp'].iloc[0]) / (len(recorded_data) - 1)
            for i in range(1, forecast_periods + 1):
                forecast_timestamps.append((last_timestamp + i * avg_timedelta).strftime('%Y-%m-%d %H:%M:%S'))
        
        results['timestamps'] = forecast_timestamps
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({"error": str(e)})