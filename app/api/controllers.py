from flask import Blueprint, jsonify, request
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
import json
from io import StringIO
import traceback # Import traceback for better error logging

# Load initial data
data = """Timestamp	ph	Turbidity	Conductivity	temperature	predicted_potability
01/04/2025	6.511618075	4.850433708	475.3413507	23.63438487	0
02/04/2025	2.803563057	3.489938597	447.594219	24.36289103	0
03/04/2025	7.818274839	3.329243166	359.09919	25.06415641	0
04/04/2025	6.898315041	3.822307139	392.3823257	27.90885435	0
05/04/2025	7.95145117	3.886882386	283.8772795	23.60588527	1
06/04/2025	6.272474991	4.61253578	448.5682598	23.59194575	1
07/04/2025	5.15839641	4.645852435	331.5867258	25.22619337	0
08/04/2025	9.313465893	4.493612765	267.8017107	28.25679184	0
09/04/2025	5.869736755	4.689315048	465.6164808	25.64961337	0
10/04/2025	6.978290868	4.931878079	564.6475642	28.57149585	0
11/04/2025	6.876451043	4.545340927	349.1860855	27.29446123	1
12/04/2025	10.30847766	3.285077052	382.9515048	27.80725071	0
13/04/2025	8.610963032	4.614400405	342.2052524	25.39744057	1
14/04/2025	6.013160966	3.455622541	444.276635	23.18407922	0
15/04/2025	4.349439999	4.442711502	432.642307	23.71855471	0
16/04/2025	7.828740105	5.199446636	290.1186126	26.25486466	1
17/04/2025	6.427365144	3.568823153	332.6127519	23.77025845	0
18/04/2025	4.692196563	4.009006062	406.551667	26.0681873	0
19/04/2025	3.637170625	5.446565703	509.1503233	26.06424022	1
20/04/2025	5.018132384	4.286478967	467.2180731	24.12398396	0
21/04/2025	5.983731294	3.84938277	443.543934	24.61870208	0
22/04/2025	7.986018499	3.153004728	434.9545189	24.7255117	1
23/04/2025	7.851925692	5.204044178	505.6656389	25.82347712	1
24/04/2025	4.477091714	4.32768029	456.9797718	25.68185373	0
25/04/2025	8.157518083	4.469430538	511.5876641	25.20731238	0
26/04/2025	10.20064451	5.174719759	291.6002828	23.51983971	0
27/04/2025	7.042793877	3.965647067	504.9254661	25.63440359	0
28/04/2025	8.383761691	3.585736635	364.7948827	26.44526897	0
29/04/2025	8.758824521	3.527565359	518.3855456	24.94206652	0
30/04/2025	7.822257363	3.080865919	593.4836023	25.8965962	0
01/05/2025	7.710137891	3.898670803	437.5140661	25.64620891	0
02/05/2025	6.931470427	4.249333174	398.6479443	26.5221263	0
03/05/2025	8.268307592	4.405408198	294.2980647	28.61081418	0
04/05/2025	6.733493842	3.178003605	397.6849857	26.67086419	0
05/05/2025	5.097786226	4.829323364	445.5626438	24.31125572	0
06/05/2025	6.435722717	4.925339107	508.345207	28.82480208	1
07/05/2025	5.368125754	2.004142586	435.6664692	25.18573512	0
"""

# ——— 2. Read into DataFrame, parsing dates with day-first ———
# Use pd.read_csv with StringIO to read the string data as if it were a file
recorded_data = pd.read_csv(
    StringIO(data),
    sep='\t',
    parse_dates=['Timestamp'],
    dayfirst=True
)

# ——— 3. Ensure proper dtypes ———
# Get the target dtypes from the initial data load
target_dtypes = {
    'ph': float,
    'Turbidity': float,
    'Conductivity': float,
    'temperature': float,
    'predicted_potability': int
}
recorded_data = recorded_data.astype(target_dtypes)


api_bp = Blueprint('api', __name__)

# Load models - ensure these paths are correct relative to where your app runs
try:
    model = joblib.load('utils/potability model.model')
    scaler = joblib.load('utils/feature_scaler.pkl')
except FileNotFoundError as e:
    print(f"Error loading model or scaler: {e}")
    print("Please ensure 'utils/potability model.model' and 'utils/feature_scaler.pkl' exist.")
    # In a real app, you might want to exit or handle this more gracefully
    model = None
    scaler = None
except Exception as e:
    print(f"An unexpected error occurred loading models: {e}")
    model = None
    scaler = None


@api_bp.route('/status')
def api_status():
    return jsonify(status='ok')

@api_bp.route('/add_data', methods=['POST'], strict_slashes=False)
def read_data():
    global recorded_data # Declare intent to modify the global variable

    try:
        # Get the raw body as bytes, then decode to string
        raw_data = request.get_data(as_text=True)

        # parse data as JSON
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid JSON'}), 400

        # Extract and validate values
        # Use .get() with a default to avoid KeyError if keys are missing
        try:
            ph_value = float(data.get('ph', 0))
            turbidity_level = float(data.get('turbidity', 0))
            conductivity = float(data.get('conductivity', 0))
            temperature = float(data.get('temperature', 0))
        except (ValueError, TypeError) as e:
             return jsonify({'error': f'Invalid data type for one or more parameters: {e}'}), 400


        timestamp = datetime.now()
        predicted_potability = 0 # Default value before ML prediction

        # validate the ph value (optional, but good practice)
        # Added checks to ensure valid float output
        if ph_value > 14:
            ph_value = float(np.random.uniform(11, 14)) # Ensure float
        elif ph_value < 0:
            ph_value = float(np.random.uniform(0, 4)) # Ensure float

        # Prepare the new row as a dictionary
        new_row_dict = {
            'Timestamp': timestamp,
            'ph': ph_value,
            'Turbidity': turbidity_level,
            'Conductivity': conductivity,
            'temperature': temperature,
            'predicted_potability': predicted_potability
        }

        # Create a DataFrame for the new row
        new_df = pd.DataFrame([new_row_dict])

        # Ensure columns and their order match recorded_data
        # This also handles potential missing columns in new_df by adding them with NaN
        if recorded_data is None or recorded_data.empty:
             print("Error: recorded_data is not initialized correctly.")
             return jsonify({"error": "Internal server error: Data storage not initialized."}), 500


        new_df = new_df.reindex(columns=recorded_data.columns)

        new_df = new_df.astype(target_dtypes)

        recorded_data = pd.concat([recorded_data, new_df], ignore_index=True)

        return jsonify({
            'ph': ph_value,
            'turbidity': turbidity_level,
            'conductivity': conductivity,
            'temperature': temperature,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S')
        })

    except Exception as e:
        # Log the full traceback for debugging
        error_trace = traceback.format_exc()
        print(f"Error in /add_data: {error_trace}")
        return jsonify({'error': f'An internal server error occurred: {str(e)}', 'traceback': error_trace}), 500

@api_bp.route('/ml_model', methods=['POST'], strict_slashes=False)
def ml_model():
    global recorded_data # Ensure we modify the global variable

    if model is None or scaler is None:
        return jsonify({"error": "ML model or scaler not loaded correctly. Cannot perform prediction."}), 500

    try:
        # Check if there's data to process
        if recorded_data is None or recorded_data.empty:
            return jsonify({"message": "No data recorded yet to process."}), 200 # Or 404/400 depending on desired behavior

        # Make a copy to avoid modifying recorded_data while dropping/scaling
        data_copy = recorded_data.copy()

        # Check for required columns before dropping
        required_features = ['ph', 'Conductivity', 'Turbidity']
        if not all(f in data_copy.columns for f in required_features):
             missing = [f for f in required_features if f not in data_copy.columns]
             return jsonify({"error": f"Missing required features for ML prediction: {missing}"}), 400


        # Select features for scaling and prediction
        features_for_scaling = data_copy[required_features]

        # Scale the data
        # The scaler was fitted on specific columns; ensure we scale the corresponding columns
        scaled_data = scaler.transform(features_for_scaling)

        # Predict the potability probabilities
        # model.predict_proba returns shape (n_samples, n_classes)
        # We usually want the probability of the positive class (e.g., class 1)
        # Assuming class 1 is 'potable' and is the second column (index 1)
        if hasattr(model, 'predict_proba'):
            predicted_potability_proba = model.predict_proba(scaled_data)[:, 1] # Probability of class 1
        elif hasattr(model, 'predict'):
             # If model only has predict (e.g., SVC without probability=True), use predict
             # Note: This gives 0 or 1, not a probability
             predicted_potability_proba = model.predict(scaled_data)
             # Convert to float for consistency if model predicts int
             predicted_potability_proba = predicted_potability_proba.astype(float)
        else:
             return jsonify({"error": "Loaded model does not have predict_proba or predict method."}), 500


        # Update the global DataFrame with the predicted potability probabilities
        # Ensure the length matches. Should match since we're processing the whole dataframe.
        if len(predicted_potability_proba) != len(recorded_data):
             # This indicates a significant issue in data processing
             print(f"Length mismatch: Predictions ({len(predicted_potability_proba)}) vs Data ({len(recorded_data)})")
             return jsonify({"error": "Internal error: Length mismatch during prediction update."}), 500

        recorded_data['predicted_potability'] = predicted_potability_proba

        # Include the predicted potability (now probability) in the response
        # Convert Timestamp to string for JSON serialization
        response_data = recorded_data.copy()
        response_data['Timestamp'] = response_data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        return jsonify(response_data.to_dict(orient='records'))

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in /ml_model: {error_trace}")
        return jsonify({'error': f'An internal server error occurred: {str(e)}', 'traceback': error_trace}), 500


@api_bp.route('/forecast', methods=["POST"], strict_slashes=False)
def forecast():
    global recorded_data

    try:
        if recorded_data is None or recorded_data.empty:
            return jsonify({"error": "No data available for forecasting. The 'recorded_data' DataFrame is empty or not initialized."}), 400

        # Ensure Timestamp is datetime and set as index for time series analysis
        # Make a copy to avoid modifying the global DataFrame's index persistently
        forecast_data = recorded_data.copy()

        if 'Timestamp' not in forecast_data.columns or not pd.api.types.is_datetime64_any_dtype(forecast_data['Timestamp']):
            return jsonify({"error": "Timestamp column is missing or not in datetime format in recorded_data."}), 400

        # Use the Timestamp column as index
        # Sort by Timestamp just in case
        forecast_data = forecast_data.sort_values('Timestamp').set_index('Timestamp')

        if len(forecast_data) < 10: # ARIMA needs a reasonable number of data points
            return jsonify({"error": f"Not enough data for forecasting. Need at least 10 data points, got {len(forecast_data)}."}), 400

        forecast_periods = request.json.get('periods', 5)
        if not isinstance(forecast_periods, int) or forecast_periods <= 0:
            return jsonify({"error": "Invalid 'periods' value. Must be a positive integer."}), 400

        results_dict = {}
        # Include temperature in forecasting
        features_to_forecast = ['ph', 'Turbidity', 'Conductivity', 'temperature', 'predicted_potability'] # Added temperature

        # Ensure all features exist
        for feature in features_to_forecast:
            if feature not in forecast_data.columns:
                 return jsonify({"error": f"Feature '{feature}' not found in recorded data."}), 400

        for feature in features_to_forecast:
            series = forecast_data[feature] # Use the series directly

            # ARIMA model can fail with constant series or other issues.
            try:
                # Use a simple order (p,d,q). (1,1,1) is a common starting point.
                # For robustness, especially with short or constant series, consider auto_arima
                # or adding checks for constant/low variance series.
                # ARIMA(0,0,0) is just a constant prediction (the mean)
                if series.nunique() <= 1:
                    # Handle constant series explicitly
                    print(f"Feature '{feature}' is constant or has only one unique value. Forecasting mean.")
                    forecast_values = np.full(forecast_periods, series.iloc[-1] if not series.empty else 0) # Forecast last value or 0 if empty
                else:
                    # Attempt ARIMA
                    # Using suppress warnings as ARIMA fit can be noisy for simple models
                    import warnings
                    with warnings.catch_warnings():
                         warnings.filterwarnings("ignore") # Ignore convergence warnings etc.
                         model = ARIMA(series, order=(1,1,1), enforce_stationarity=False, enforce_invertibility=False)
                         print(f"Fitting ARIMA model for {feature}...")
                         model_fit = model.fit()
                         forecast_values = model_fit.forecast(steps=forecast_periods)

                # ARIMA forecast returns a NumPy array. Convert to list of native Python floats.
                # Use a dictionary with string keys '0', '1', ...
                results_dict[feature] = {str(i): float(val) for i, val in enumerate(forecast_values)}

            except Exception as e:
                # Log the specific error for the feature
                print(f"Error fitting or forecasting ARIMA for {feature}: {e}")
                # Indicate failure for this specific feature forecast
                results_dict[feature] = {"error": f"Could not forecast {feature}: {str(e)}"}


        # Generate future timestamps based on the last timestamp and average interval
        forecast_timestamps_ms = {}
        if not forecast_data.empty:
            last_timestamp_dt = forecast_data.index[-1] # Get the last timestamp from the index

            # Determine time interval
            if len(forecast_data) > 1:
                 # Calculate the difference between the last two timestamps as the interval
                 time_interval = forecast_data.index[-1] - forecast_data.index[-2]
                 # If interval is zero (duplicate timestamps), default to a small value
                 if time_interval.total_seconds() <= 0:
                     print("Warning: Duplicate or non-increasing timestamps found. Using 1-day interval.")
                     time_interval = timedelta(days=1) # Default to 1 day if interval is zero or negative
            else:
                 # If only one data point, default to a reasonable interval (e.g., 1 day)
                 print("Warning: Only one data point available. Using 1-day interval for forecast timestamps.")
                 time_interval = timedelta(days=1) # Default to 1 day if only one point


            for i in range(forecast_periods):
                # Calculate next timestamp by adding the determined interval
                next_timestamp_dt = last_timestamp_dt + (i + 1) * time_interval
                # Convert to milliseconds since epoch for JavaScript Date compatibility
                forecast_timestamps_ms[str(i)] = int(next_timestamp_dt.timestamp() * 1000)
        else:
             # Fallback if somehow forecast_data is empty (should have been caught earlier)
            print("Warning: No data available to generate forecast timestamps.")
            current_time_ms = int(datetime.now().timestamp() * 1000)
            for i in range(forecast_periods):
                # Default timestamp interval (e.g., 1 day = 86400000 ms)
                forecast_timestamps_ms[str(i)] = current_time_ms + (i + 1) * 86400000


        results_dict['timestamps'] = forecast_timestamps_ms

        return jsonify(results_dict)

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"An unexpected error occurred in /forecast: {error_trace}")
        return jsonify({"error": f"An internal server error occurred: {str(e)}", "traceback": error_trace}), 500

@api_bp.route('/get_data', methods=['GET'], strict_slashes=False)
def get_data():
    global recorded_data # Access the global variable

    try:
        if recorded_data is None or recorded_data.empty:
             return jsonify([]), 200 # Return empty list if no data

        # Make a copy and convert Timestamp to string for JSON serialization
        data_copy = recorded_data.copy()
        # Check if Timestamp column exists and is datetime-like before formatting
        if 'Timestamp' in data_copy.columns and pd.api.types.is_datetime64_any_dtype(data_copy['Timestamp']):
             data_copy['Timestamp'] = data_copy['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        # else: Timestamp is already in a JSON-serializable format or not present, do nothing

        # Convert the DataFrame to a list of dictionaries
        data = data_copy.to_dict(orient='records')

        # Return the data as JSON
        return jsonify(data)

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in /get_data: {error_trace}")
        # Corrected the return statement in the except block
        return jsonify({'error': f'An internal server error occurred: {str(e)}', 'traceback': error_trace}), 500