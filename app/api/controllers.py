from flask import Blueprint, jsonify, request
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
import json
from io import StringIO


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
recorded_data = pd.read_csv(
    StringIO(data),
    sep='\t',
    parse_dates=['Timestamp'],
    dayfirst=True
)

# ——— 3. Ensure proper dtypes ———
recorded_data = recorded_data.astype({
    'ph': float,
    'Turbidity': float,
    'Conductivity': float,
    'temperature': float,
    'predicted_potability': int
})


api_bp = Blueprint('api', __name__)
model = joblib.load('utils/potability model.model')
scaler = joblib.load('utils/feature_scaler.pkl')


@api_bp.route('/status')
def api_status():
    return jsonify(status='ok')

@api_bp.route('/add_data', methods=['POST'], strict_slashes=False)
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
        ph_value = float(data.get('ph', 0))
        turbidity_level = float(data.get('turbidity', 0))
        conductivity = float(data.get('conductivity', 0))
        temperature = float(data.get('temperature', 0))
        timestamp = datetime.now()
        predicted_potability = 0
        
        # validate the ph value, if it is greater than 14, change the value to a random value be between 10 and 14, if it is less than 0, change the value to a random value between 0 and 4
        if ph_value > 14:
            ph_value = np.random.uniform(11, 14)
        elif ph_value < 0:
            ph_value = np.random.uniform(0, 4)

        new_row = {
            'Timestamp': timestamp,
            'ph': ph_value,
            'Turbidity': turbidity_level,
            'Conductivity': conductivity,
            'temperature': temperature,
            'predicted_potability': predicted_potability
        }


        global recorded_data
        recorded_data = pd.concat([recorded_data, pd.DataFrame([new_row])], ignore_index=True)
        print("Recorded Global data received:", recorded_data)

        # now emit the socket event to read data in real time
        # socketio.emit('new_data', {
        #     'ph': ph,
        #     'turbidity': turbidity,
        #     'conductivity': conductivity,
        #     'temperature': temperature,
        #     'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S')
        # })

        return jsonify({
            'ph': ph_value,
            'turbidity': turbidity_level,
            'conductivity': conductivity,
            'temperature': temperature,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/ml_model', methods=['POST'], strict_slashes=False)
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
        predicted_potability = model.predict_proba(data)
        
        # Update the global DataFrame with the predicted potability
        recorded_data['predicted_potability'] = predicted_potability
        
        # Include the predicted potability in the response
        response = recorded_data.to_dict(orient='records')
        return jsonify(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(error=str(e))


@api_bp.route('/forecast', methods=["POST"], strict_slashes=False)
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

@api_bp.route('/get_data', methods=['GET'], strict_slashes=False)
def get_data():
    try:
        # Convert the DataFrame to a list of dictionaries
        data = recorded_data.to_dict(orient='records')
        
        # Return the data as JSON
        return jsonify(data)
    
    except Exception as e:
        return jsonify(error=str(e))