# src/api/server_improved.py

from flask import Flask, request, jsonify
from flask.json.provider import DefaultJSONProvider
import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import traceback
from collections import deque
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import the feature engineering module
sys.path.append(str(Path(__file__).parent.parent / 'ml'))
from feature_engineering import AirQualityFeatureEngineer

# Custom JSON provider to handle NumPy types
class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app = Flask(__name__)
app.json_provider_class = NumpyJSONProvider
app.json = app.json_provider_class(app)

# Global variables for model and historical data
model = None
feature_engineer = None
historical_data = deque(maxlen=100)  # Keep last 100 measurements

def initialize_system():
    """Initialize model, feature engineer, and historical data"""
    global model, feature_engineer, historical_data
    
    model_path = Path(__file__).parent.parent.parent / 'models' / 'artifacts' / 'best_model.pkl'
    feature_engineer_path = Path(__file__).parent.parent.parent / 'models' / 'artifacts' / 'feature_engineer.pkl'
    
    # Load model
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
    
    # Load feature engineer
    try:
        with open(feature_engineer_path, 'rb') as f:
            feature_engineer = pickle.load(f)
        print(f"Feature engineer loaded successfully from {feature_engineer_path}")
    except Exception as e:
        print(f"Error loading feature engineer: {e}")
        feature_engineer = None
    
    # Initialize historical data with realistic values
    initialize_historical_data()

def initialize_historical_data():
    """Initialize with synthetic historical data"""
    global historical_data
    current_time = datetime.now()
    
    # Create realistic historical data based on typical values
    for i in range(100):
        time_delta = timedelta(hours=i)
        timestamp = current_time - time_delta
        
        # Generate realistic values with daily patterns
        hour = (timestamp.hour) % 24
        
        # CO values typically vary with time of day
        co_base = 2.0
        co_variation = np.sin(2 * np.pi * hour / 24) * 0.8 + np.random.normal(0, 0.2)
        
        data_point = {
            'DateTime': timestamp,
            'CO(GT)': max(0.1, co_base + co_variation),
            'PT08.S1(CO)': 1300 + np.random.normal(0, 100),
            'NMHC(GT)': 150 + np.random.normal(0, 20),
            'C6H6(GT)': 10 + np.random.normal(0, 2),
            'PT08.S2(NMHC)': 1000 + np.random.normal(0, 100),
            'NOx(GT)': 160 + np.random.normal(0, 30),
            'PT08.S3(NOx)': 1000 + np.random.normal(0, 100),
            'NO2(GT)': 110 + np.random.normal(0, 20),
            'PT08.S4(NO2)': 1600 + np.random.normal(0, 200),
            'PT08.S5(O3)': 1200 + np.random.normal(0, 150),
            'T': 15 + np.random.normal(0, 5),
            'RH': 50 + np.random.normal(0, 10),
            'AH': 0.8 + np.random.normal(0, 0.2)
        }
        historical_data.appendleft(data_point)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "feature_engineer_loaded": feature_engineer is not None,
        "historical_data_size": len(historical_data)
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or feature_engineer is None:
        return jsonify({
            "error": "Model or feature engineer not loaded",
            "status": "error"
        }), 500

    try:
        data = request.get_json()
        
        # Add to historical data
        data_with_timestamp = data.copy()
        if 'DateTime' not in data_with_timestamp:
            data_with_timestamp['DateTime'] = datetime.now()
        else:
            data_with_timestamp['DateTime'] = pd.to_datetime(data_with_timestamp['DateTime'])
        
        historical_data.append(data_with_timestamp)
        
        # Create DataFrame from historical data
        hist_df = pd.DataFrame(list(historical_data))
        hist_df['DateTime'] = pd.to_datetime(hist_df['DateTime'])
        
        # Set historical data in feature engineer
        feature_engineer.historical_data = hist_df
        
        # Prepare features using historical data
        df_features = feature_engineer.prepare_single_prediction(data, use_historical_data=True)
        
        # Extract feature columns
        X = df_features[feature_engineer.feature_names]
        
        # Scale features
        X_scaled = feature_engineer.scale_features(X)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        
        return jsonify({
            "prediction": float(prediction),
            "status": "success",
            "historical_data_used": True,
            "features_used": len(feature_engineer.feature_names)
        })

    except Exception as e:
        tb = traceback.format_exc()
        print("Error in prediction:", e, tb)
        return jsonify({
            "error": str(e),
            "status": "error",
            "traceback": tb
        }), 500

# Initialize the system when the server starts
initialize_system()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)