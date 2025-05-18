# import required libraries
from flask import Blueprint, jsonify, request
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
# Predict future values based on past observations
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
import json
from app import socketio

# Initiate empty dataframe to store water quality data and predictions
recorded_data = pd.DataFrame(columns=[
                             'Timestamp', 'ph', 'Turbidity', 'Conductivity', 'temperature', 'predicted_potability'])

# Create a flask blueprint for API routes
api_bp = Blueprint('api', __name__)

# Load the pre-trained machine learning model and scaler
model = joblib.load('utils/potability model.model')
scaler = joblib.load('utils/feature_scaler.pkl')

# Basic API status check endpoint


@api_bp.route('/status')
def api_status():
    # Return a simple JSON response indicating the API is working
    return jsonify(status='ok')

# Basic API status check endpoint


@api_bp.route('/add_data', methods=['POST'])
def read_data():
    try:
        # Get the raw body as bytes, then decode to string
        raw_data = request.get_data(as_text=True)

        # parse data as JSON if you expect it to be JSON
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid JSON'}), 400

        # Extract values from JSON, use default 0 if key is missing
        ph = float(data.get('ph', 0))
        turbidity = float(data.get('turbidity', 0))
        conductivity = float(data.get('conductivity', 0))
        temperature = float(data.get('temperature', 0))
        timestamp = datetime.now()  # Record the current timestamp
        predicted_potability = 0  # Default potability before prediction

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
        print("Recorded Global data received:", recorded_data)

        # now emit the socket event to read data in real time
        socketio.emit('new_data', {
            'ph': ph,
            'turbidity': turbidity,
            'conductivity': conductivity,
            'temperature': temperature,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S')
        })

        # Respond with the received data and timestamp
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


@api_bp.route('/ml_model', methods=['POST'])
def ml_model():
    try:
        # Load a copy of the recorded data
        global recorded_data
        data = recorded_data.copy()

        # Drop the timestamp column
        data = data.drop(
            columns=['Timestamp', 'predicted_potability', 'temperature'])

        # scale the relevant features using the pre-fitted scaler
        data = scaler.transform(data[['ph', 'Conductivity', 'Turbidity']])

        # Use the loaded model to predict water potability
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

# Endpoint to forecast future values using ARIMA model


@api_bp.route('/forecast', methods=["POST"])
def forecast():
    try:
        global recorded_data  # Make sure this DataFrame is populated and 'Timestamp' is datetime

        # Ensure there is enough data to run forecasting
        if not isinstance(recorded_data, pd.DataFrame) or recorded_data.empty:
            # Validate that recorded_data exists and is not empty
            return jsonify({"error": "No data available for forecasting. The 'recorded_data' DataFrame is empty or not initialized."}), 400

        if 'Timestamp' not in recorded_data.columns or not pd.api.types.is_datetime64_any_dtype(recorded_data['Timestamp']):
            # Check if the 'Timestamp' column exists and is in a datetime format
            return jsonify({"error": "Timestamp column is missing or not in datetime format in recorded_data."}), 400

        if len(recorded_data) < 10:
            # Ensure enough data points for a meaningful forecast
            return jsonify({"error": f"Not enough data for forecasting. Need at least 10 data points, got {len(recorded_data)}."}), 400

        # Get forecast periods from the request (defaults to 5 if not provided)
        forecast_periods = request.json.get('periods', 5)
        if not isinstance(forecast_periods, int) or forecast_periods <= 0:
            # Validate that the forecast period is a positive integer
            return jsonify({"error": "Invalid 'periods' value. Must be a positive integer."}), 400

        # Initialize dictionary to store forecast results
        results_dict = {}

        # List of features to forecast
        features_to_forecast = [
            'ph', 'Turbidity', 'Conductivity', 'temperature', 'predicted_potability']

        # Check that each feature exists in the DataFrame
        for feature in features_to_forecast:
            if feature not in recorded_data.columns:
                return jsonify({"error": f"Feature '{feature}' not found in recorded data."}), 400

        # Forecast each feature using ARIMA model
        for feature in features_to_forecast:
            series = recorded_data[feature].values

            # ARIMA model can fail with constant series or other issues.
            try:
                # Using a simple order; consider auto_arima or specific orders per feature
                # SARIMAX might be better if seasonality is present.
                model = ARIMA(series, order=(
                    1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
                print(f"Fitting ARIMA model for {feature}...")
                model_fit = model.fit()
                forecast_values = model_fit.forecast(steps=forecast_periods)
                # ARIMA forecast returns a NumPy array. Convert to list of native Python floats.
                # Convert to dictionary with string keys '0', '1', ...
                results_dict[feature] = {str(i): float(
                    val) for i, val in enumerate(forecast_values)}
            except Exception as e:
                # Catch model fitting failures (e.g., constant data)
                print(f"Error fitting ARIMA for {feature}: {e}")
                return jsonify({"error": f"Could not forecast {feature}. ARIMA model fitting failed: {str(e)}"}), 500

        # Generate future timestamps
        # Timestamps should be in milliseconds since epoch for JavaScript Date compatibility
        forecast_timestamps_ms = {}
        if not recorded_data['Timestamp'].empty:
            # This is a pandas Timestamp object
            last_timestamp_dt = recorded_data['Timestamp'].iloc[-1]

            avg_timedelta_seconds = 0
            if len(recorded_data) > 1:
                # Calculate average difference in seconds
                time_diffs = recorded_data['Timestamp'].diff().dropna()
                if not time_diffs.empty:
                    avg_timedelta_seconds = time_diffs.mean().total_seconds()
                else:  # only one historical point after diff
                    avg_timedelta_seconds = 20  # Default to 20s if only one diff possible or no diff
            else:  # only one historical point
                avg_timedelta_seconds = 20  # Default to 20s if only one point

            if avg_timedelta_seconds <= 0:  # Ensure positive timedelta
                avg_timedelta_seconds = 20

            # Generate future timestamps using average interval
            for i in range(forecast_periods):
                # Calculate next timestamp by adding average timedelta
                next_timestamp_dt = last_timestamp_dt + \
                    timedelta(seconds=(i + 1) * avg_timedelta_seconds)
                # Convert to milliseconds since epoch
                forecast_timestamps_ms[str(i)] = int(
                    next_timestamp_dt.timestamp() * 1000)  # Convert to milliseconds
        else:
            # Fallback if somehow timestamps are empty (should have been caught earlier)
            current_time_ms = int(pd.Timestamp.now().timestamp() * 1000)
            for i in range(forecast_periods):
                # Default to 20s interval
                forecast_timestamps_ms[str(
                    i)] = current_time_ms + (i + 1) * 20000

        results_dict['timestamps'] = forecast_timestamps_ms

        return jsonify(results_dict)

    # Global error handler incase something unexpected happens
    except Exception as e:
        print(f"An unexpected error occurred in /forecast: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

# Endpoint to get all recorded data


@api_bp.route('/get_data', methods=['GET'])
def get_data():
    try:
        # Convert the DataFrame to a list of dictionaries
        data = recorded_data.to_dict(orient='records')

        # Return all the data as JSON array
        return jsonify(data)

    except Exception as e:
        # Handle any errors that occur during during the conversion or access
        return jsonify(error=str(e))
