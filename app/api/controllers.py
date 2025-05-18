# Import required libraries
from flask import Blueprint, jsonify, request
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
# predict future values based on past observations
from statsmodels.tsa.arima.model import ARIMA


# Initiate empty dataframe to store water quality data and predictions
recorded_data = pd.DataFrame(columns=[
                             'Timestamp', 'ph', 'Turbidity', 'Conductivity', 'temperature', 'predicted_potability'])

# Create a Flask blueprint for API routes
api_bp = Blueprint('api', __name__)

# Load the pre-trained machine learning model and scaler
model = joblib.load('utils/potability model.model')
scaler = joblib.load('utils/feature_scaler.pkl')

# Basic API status check endpoint


@api_bp.route('/status')
def api_status():
    # Return a simple JSON response indicating the API is working
    return jsonify(status='ok')

# Endpoint to accept new water quality readings


@api_bp.route('/add_data', methods=['POST'])
def read_data():
    try:
        # Get data from JSON request body
        data = request.get_json()

        # Extract values from JSON, use default 0 if kwy is miss
        ph = float(data.get('ph', 0))
        turbidity = float(data.get('turbidity', 0))
        conductivity = float(data.get('conductivity', 0))
        temperature = float(data.get('temperature', 0))
        timestamp = datetime.now()
        predicted_potability = 0

        # Create a new row with the provided and default values
        new_row = {
            'Timestamp': timestamp,
            'ph': ph,
            'Turbidity': turbidity,
            'Conductivity': conductivity,
            'temperature': temperature,
            'predicted_potability': predicted_potability
        }

        # Add the new row to the global recorded_data Dataframe
        global recorded_data
        recorded_data = pd.concat(
            [recorded_data, pd.DataFrame([new_row])], ignore_index=True)

        # Prepare a JSON response that includes a timestamp
        return jsonify({
            'ph': ph,
            'turbidity': turbidity,
            'conductivity': conductivity,
            'temperature': temperature,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S')
        })

    except Exception as e:
        # Return error message if something goes wrong
        return jsonify({'error': str(e)}), 500

 # Endpoint to make predictions using the ML model


@api_bp.route('/ml_model', methods=['POST'])
def ml_model():
    try:
        # Load the data from the global DataFrame
        global recorded_data
        data = recorded_data.copy()

        # Drop the timestamp column which are non-features before prediction
        data = data.drop(
            columns=['Timestamp', 'predicted_potability', 'temperature'])

        # scale the relevant features using the pre-fitted scaler
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
        traceback.print_exc()  # Print traceback to console for debugging
        return jsonify(error=str(e))


# Endpoint to forecast future values using ARIMA model
@api_bp.route('/forecast', methods=["POST"])
def forecast():
    try:
        global recorded_data

        # Ensure there is enough data to run forecasting
        if len(recorded_data) < 10:  # Need sufficient data for forecasting
            return jsonify({"error": "Not enough data for forecasting. Need at least 10 data points."})

        # Number of time periods to forecast into future results
        forecast_periods = request.json.get('periods', 5)

        results = {}  # Dictionary to store forecasted results

        # List of features to forecast
        features = ['ph', 'Turbidity', 'Conductivity',
                    'temperature', 'predicted_potability']

        for feature in features:
            # Convert data to time series
            # Extract the time series data for the feature
            series = recorded_data[feature].values

            # Fit ARIMA model - using simple (1,1,1) parameters
            # In production, you might want to use auto_arima to find optimal parameters
            model = ARIMA(series, order=(1, 1, 1))
            model_fit = model.fit()

            # Make forecast for future values
            forecast = model_fit.forecast(steps=forecast_periods)

            # Store forecasted values in te dictionary
            results[feature] = forecast.tolist()

        # Add timestamp information to the results
        last_timestamp = recorded_data['Timestamp'].iloc[-1]
        forecast_timestamps = []

        # Generate future timestamps based on average time difference
        if len(recorded_data) > 1:
            # Calculate average time difference between records
            avg_timedelta = (recorded_data['Timestamp'].iloc[-1] -
                             recorded_data['Timestamp'].iloc[0]) / (len(recorded_data) - 1)
            # Generate future timestamps based on average interval
            for i in range(1, forecast_periods + 1):
                forecast_timestamps.append(
                    (last_timestamp + i * avg_timedelta).strftime('%Y-%m-%d %H:%M:%S'))

        results['timestamps'] = forecast_timestamps

        return jsonify(results)

    # Return error if forecasting fails
    except Exception as e:
        return jsonify({"error": str(e)})

# End point to retrieve all recorded data (including predictions)


@api_bp.route('/get_data', methods=['GET'])
def get_data():
    try:
        # Convert the DataFrame to a list of dictionaries
        data = recorded_data.to_dict(orient='records')

        # Return the data as JSON
        return jsonify(data)

    except Exception as e:
        # Handle any unexpected error
        return jsonify(error=str(e))
