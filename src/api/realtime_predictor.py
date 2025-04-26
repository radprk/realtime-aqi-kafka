# src/api/realtime_predictor.py

import pandas as pd
import numpy as np
import pickle
from collections import deque
from datetime import datetime, timedelta
import sys
from pathlib import Path

class RealtimePredictor:
    """Real-time prediction system with proper historical data handling"""
    
    def __init__(self, window_size=100):
        """Initialize predictor with historical data storage"""
        self.window_size = window_size
        self.historical_data = deque(maxlen=window_size)
        
        # Load model
        project_root = Path(__file__).parent.parent.parent
        self.model_path = project_root / 'models' / 'artifacts' / 'best_model.pkl'
        self.feature_engineer_path = project_root / 'models' / 'artifacts' / 'feature_engineer.pkl'
        
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load feature engineer and extract what we need
        with open(self.feature_engineer_path, 'rb') as f:
            feature_engineer_obj = pickle.load(f)
            self.feature_names = feature_engineer_obj.feature_names
            self.scaler = feature_engineer_obj.scaler
        
        # Initialize with some synthetic historical data (for demonstration)
        self._initialize_historical_data()
    
    def _initialize_historical_data(self):
        """Initialize with synthetic historical data"""
        current_time = datetime.now()
        for i in range(self.window_size):
            time_delta = timedelta(hours=i)
            timestamp = current_time - time_delta
            
            # Synthetic data - in production, load from database
            data_point = {
                'DateTime': timestamp,
                'CO(GT)': np.random.normal(2.0, 0.5),
                'PT08.S1(CO)': np.random.normal(1300, 100),
                'NMHC(GT)': np.random.normal(150, 20),
                'C6H6(GT)': np.random.normal(10, 2),
                'PT08.S2(NMHC)': np.random.normal(1000, 100),
                'NOx(GT)': np.random.normal(160, 30),
                'PT08.S3(NOx)': np.random.normal(1000, 100),
                'NO2(GT)': np.random.normal(110, 20),
                'PT08.S4(NO2)': np.random.normal(1600, 200),
                'PT08.S5(O3)': np.random.normal(1200, 150),
                'T': np.random.normal(15, 5),
                'RH': np.random.normal(50, 10),
                'AH': np.random.normal(0.8, 0.2)
            }
            self.historical_data.append(data_point)
    
    def add_measurement(self, data):
        """Add new measurement to historical data"""
        self.historical_data.append(data)
    
    def create_temporal_features(self, df):
        """Create temporal features"""
        df['hour'] = df['DateTime'].dt.hour
        df['day'] = df['DateTime'].dt.day
        df['month'] = df['DateTime'].dt.month
        df['dayofweek'] = df['DateTime'].dt.dayofweek
        df['quarter'] = df['DateTime'].dt.quarter
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        
        # Cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def predict(self, data):
        """Make prediction using proper historical context"""
        # Add current measurement to historical data
        self.add_measurement(data)
        
        # Convert historical data to DataFrame
        hist_df = pd.DataFrame(list(self.historical_data))
        
        # Ensure DateTime is properly formatted
        hist_df['DateTime'] = pd.to_datetime(hist_df['DateTime'])
        
        # Create temporal features
        hist_df = self.create_temporal_features(hist_df)
        
        # Create lag features
        target_col = 'CO(GT)'
        for lag in [1, 3, 6, 12, 24]:
            hist_df[f'{target_col}_lag_{lag}'] = hist_df[target_col].shift(lag)
        
        # Create rolling features
        for window in [3, 6, 12, 24]:
            hist_df[f'{target_col}_rolling_mean_{window}'] = hist_df[target_col].rolling(window=window).mean()
            hist_df[f'{target_col}_rolling_std_{window}'] = hist_df[target_col].rolling(window=window).std()
            hist_df[f'{target_col}_rolling_min_{window}'] = hist_df[target_col].rolling(window=window).min()
            hist_df[f'{target_col}_rolling_max_{window}'] = hist_df[target_col].rolling(window=window).max()
        
        # Create EMA features
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
            hist_df[f'{target_col}_ema_{alpha}'] = hist_df[target_col].ewm(alpha=alpha).mean()
        
        # Create interaction features
        pollutant_cols = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 
                        'PT08.S4(NO2)', 'PT08.S5(O3)']
        
        for i, col1 in enumerate(pollutant_cols):
            for col2 in pollutant_cols[i+1:]:
                if col1 in hist_df.columns and col2 in hist_df.columns:
                    hist_df[f'{col1}_x_{col2}'] = hist_df[col1] * hist_df[col2]
                    hist_df[f'{col1}_div_{col2}'] = hist_df[col1] / (hist_df[col2] + 1e-8)
        
        # Get the latest row
        latest_row = hist_df.iloc[-1]
        
        # Extract feature columns in the correct order
        feature_values = []
        for feature in self.feature_names:
            if feature in latest_row.index:
                feature_values.append(latest_row[feature])
            else:
                # Handle missing features (this shouldn't happen if everything is set up correctly)
                feature_values.append(0.0)
        
        # Convert to numpy array
        X = np.array(feature_values).reshape(1, -1)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)[0]
        
        return prediction

# Example usage:
if __name__ == "__main__":
    predictor = RealtimePredictor()
    
    # Example prediction
    data = {
        "DateTime": datetime.now(),
        "CO(GT)": 2.6,
        "PT08.S1(CO)": 1360,
        "NMHC(GT)": 150,
        "C6H6(GT)": 11.9,
        "PT08.S2(NMHC)": 1046,
        "NOx(GT)": 166,
        "PT08.S3(NOx)": 1056,
        "NO2(GT)": 113,
        "PT08.S4(NO2)": 1692,
        "PT08.S5(O3)": 1268,
        "T": 13.6,
        "RH": 48.9,
        "AH": 0.7578
    }
    
    prediction = predictor.predict(data)
    print(f"Predicted CO(GT): {prediction}")