from flask import Blueprint, jsonify, request
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
import json
from app import socketio


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
        # Get the raw body as bytes, then decode to string
        raw_data = request.get_data(as_text=True)


        # parse data as JSON if you expect it to be JSON
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid JSON'}), 400

        # Extract values
        ph = float(data.get('ph', 0))
        turbidity = float(data.get('turbidity', 0))
        conductivity = float(data.get('conductivity', 0))
        temperature = float(data.get('temperature', 0))
        timestamp = datetime.now()
        predicted_potability = 0

        new_row = {
            'Timestamp': timestamp,
            'ph': ph,
            'Turbidity': turbidity,
            'Conductivity': conductivity,
            'temperature': temperature,
            'predicted_potability': predicted_potability
        }



        global recorded_data
        recorded_data = pd.concat([recorded_data, pd.DataFrame([new_row])], ignore_index=True)
        print("Recorded Global data received:", recorded_data)

        # now emit the socket event to read data in real time
        socketio.emit('new_data', {
            'ph': ph,
            'turbidity': turbidity,
            'conductivity': conductivity,
            'temperature': temperature,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S')
        })

        return jsonify({
            'ph': ph,
            'turbidity': turbidity,
            'conductivity': conductivity,
            'temperature': temperature,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
        global recorded_data # Make sure this DataFrame is populated and 'Timestamp' is datetime
        
        if not isinstance(recorded_data, pd.DataFrame) or recorded_data.empty:
            return jsonify({"error": "No data available for forecasting. The 'recorded_data' DataFrame is empty or not initialized."}), 400
        
        if 'Timestamp' not in recorded_data.columns or not pd.api.types.is_datetime64_any_dtype(recorded_data['Timestamp']):
            return jsonify({"error": "Timestamp column is missing or not in datetime format in recorded_data."}), 400

        if len(recorded_data) < 10:
            return jsonify({"error": f"Not enough data for forecasting. Need at least 10 data points, got {len(recorded_data)}."}), 400
        
        forecast_periods = request.json.get('periods', 5)
        if not isinstance(forecast_periods, int) or forecast_periods <= 0:
            return jsonify({"error": "Invalid 'periods' value. Must be a positive integer."}), 400

        results_dict = {}
        features_to_forecast = ['ph', 'Turbidity', 'Conductivity', 'temperature', 'predicted_potability']
        
        # Ensure all features exist in the dataframe
        for feature in features_to_forecast:
            if feature not in recorded_data.columns:
                return jsonify({"error": f"Feature '{feature}' not found in recorded data."}), 400

        for feature in features_to_forecast:
            series = recorded_data[feature].values
            
            # ARIMA model can fail with constant series or other issues.
            try:
                # Using a simple order; consider auto_arima or specific orders per feature
                # SARIMAX might be better if seasonality is present.
                model = ARIMA(series, order=(1,1,1), enforce_stationarity=False, enforce_invertibility=False)
                print(f"Fitting ARIMA model for {feature}...")
                model_fit = model.fit()
                forecast_values = model_fit.forecast(steps=forecast_periods)
                # ARIMA forecast returns a NumPy array. Convert to list of native Python floats.
                # Convert to dictionary with string keys '0', '1', ...
                results_dict[feature] = {str(i): float(val) for i, val in enumerate(forecast_values)}
            except Exception as e:
                print(f"Error fitting ARIMA for {feature}: {e}")
                return jsonify({"error": f"Could not forecast {feature}. ARIMA model fitting failed: {str(e)}"}), 500
        
        # Generate future timestamps
        # Timestamps should be in milliseconds since epoch for JavaScript Date compatibility
        forecast_timestamps_ms = {}
        if not recorded_data['Timestamp'].empty:
            last_timestamp_dt = recorded_data['Timestamp'].iloc[-1] # This is a pandas Timestamp object
            
            avg_timedelta_seconds = 0
            if len(recorded_data) > 1:
                # Calculate average difference in seconds
                time_diffs = recorded_data['Timestamp'].diff().dropna()
                if not time_diffs.empty:
                    avg_timedelta_seconds = time_diffs.mean().total_seconds()
                else: # only one historical point after diff
                    avg_timedelta_seconds = 20 # Default to 20s if only one diff possible or no diff
            else: # only one historical point
                avg_timedelta_seconds = 20 # Default to 20s if only one point

            if avg_timedelta_seconds <= 0: # Ensure positive timedelta
                avg_timedelta_seconds = 20


            for i in range(forecast_periods):
                # Calculate next timestamp by adding average timedelta
                next_timestamp_dt = last_timestamp_dt + timedelta(seconds=(i + 1) * avg_timedelta_seconds)
                # Convert to milliseconds since epoch
                forecast_timestamps_ms[str(i)] = int(next_timestamp_dt.timestamp() * 1000)
        else:
             # Fallback if somehow timestamps are empty (should have been caught earlier)
            current_time_ms = int(pd.Timestamp.now().timestamp() * 1000)
            for i in range(forecast_periods):
                forecast_timestamps_ms[str(i)] = current_time_ms + (i + 1) * 20000 # Default to 20s interval

        results_dict['timestamps'] = forecast_timestamps_ms
        
        return jsonify(results_dict)

    except Exception as e:
        print(f"An unexpected error occurred in /forecast: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

@api_bp.route('/get_data', methods=['GET'])
def get_data():
    try:
        # Convert the DataFrame to a list of dictionaries
        data = recorded_data.to_dict(orient='records')
        
        # Return the data as JSON
        return jsonify(data)
    
    except Exception as e:
        return jsonify(error=str(e))