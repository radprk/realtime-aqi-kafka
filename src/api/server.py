# src/api/server.py

from flask import Flask, request, jsonify
from flask.json.provider import DefaultJSONProvider
import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import traceback

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
# Register our custom provider
app.json_provider_class = NumpyJSONProvider
app.json = app.json_provider_class(app)

# Load model and scaler
model_path  = Path(__file__).parent.parent.parent / 'models' / 'artifacts' / 'best_model.pkl'
scaler_path = Path(__file__).parent.parent.parent / 'models' / 'artifacts' / 'scaler.pkl'

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

try:
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"Scaler loaded successfully from {scaler_path}")
except Exception as e:
    print(f"Error loading scaler: {e}")
    scaler = None

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status":        "healthy",
        "model_loaded":  model is not None,
        "scaler_loaded": scaler is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({
            "error":  "Model or scaler not loaded",
            "status": "error"
        }), 500

    try:
        data = request.get_json()
        print(f"Received payload: {data}")

        # 1) Build base dict
        df_dict = {}

        # 2) Temporal features
        if 'DateTime' in data:
            dt = pd.to_datetime(data['DateTime'])
        else:
            dt = pd.Timestamp.now()
        hour, day, month, dow = dt.hour, dt.day, dt.month, dt.dayofweek

        df_dict.update({
            'hour': hour,
            'day': day,
            'month': month,
            'dayofweek': dow,
            'hour_sin':   np.sin(2 * np.pi * hour/24),
            'hour_cos':   np.cos(2 * np.pi * hour/24),
            'dow_sin':    np.sin(2 * np.pi * dow/7),
            'dow_cos':    np.cos(2 * np.pi * dow/7),
            'month_sin':  np.sin(2 * np.pi * month/12),
            'month_cos':  np.cos(2 * np.pi * month/12),
        })

        # 3) Basic pollutant features
        for feat in [
            'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
            'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)',
            'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)',
            'T', 'RH', 'AH'
        ]:
            df_dict[feat] = data.get(feat, 0.0)

        # 4) Placeholder lag/rolling
        target_col    = 'CO(GT)'
        current_value = data.get(target_col, 0.0)
        for lag in [1,3,6,12,24]:
            df_dict[f'{target_col}_lag_{lag}'] = current_value
        for w in [3,6,12,24]:
            df_dict[f'{target_col}_rolling_mean_{w}'] = current_value
            df_dict[f'{target_col}_rolling_std_{w}']  = 0.0

        # 5) Build DataFrame
        df = pd.DataFrame([df_dict])

        # 6) Align to scaler features
        feature_cols = list(scaler.feature_names_in_)
        X = df.reindex(columns=feature_cols, fill_value=0.0).astype(float)

        print("Features prepared:", X.iloc[0].to_dict())
        X_scaled = scaler.transform(X)
        print("Scaled features shape:", X_scaled.shape)

        # 8) Predict
        prediction = model.predict(X_scaled)[0]
        print("Prediction:", prediction)

        return jsonify({
            "prediction": prediction,
            "status":     "success"
        })

    except Exception as e:
        tb = traceback.format_exc()
        print("Error in prediction:", e, tb)
        return jsonify({
            "error":     str(e),
            "status":    "error",
            "traceback": tb
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
