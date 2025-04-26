# src/api/model_evaluator.py

import requests
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import your feature engineering module directly
sys.path.append(str(Path(__file__).parent.parent / 'ml'))
from feature_engineering import AirQualityFeatureEngineer

API_URL = "http://localhost:8080"

class ModelEvaluator:
    def __init__(self):
        self.data_file = project_root / 'data' / 'raw' / 'AirQualityUCI.csv'
        self.model_path = project_root / 'models' / 'artifacts' / 'best_model.pkl'
        self.feature_engineer_path = project_root / 'models' / 'artifacts' / 'feature_engineer.pkl'
        
    def load_data(self):
        """Load and preprocess the data"""
        print("Loading original dataset...")
        df = pd.read_csv(self.data_file, sep=';', decimal=',')
        
        # Preprocess data
        numeric_columns = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 
                          'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 
                          'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 
                          'T', 'RH', 'AH']
        
        for col in numeric_columns:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace(',', '.')
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].replace(-200, np.nan)
        
        # Create DateTime column
        try:
            df['DateTime'] = pd.to_datetime(
                df['Date'] + ' ' + df['Time'],
                format='%d/%m/%Y %H.%M.%S',
                errors='coerce'
            )
        except:
            print("Error parsing dates")
        
        # Fill missing values
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].mean())
        
        return df
    
    def evaluate_with_test_data(self, n_samples=100):
        """Test model with actual data and compare predictions"""
        
        # Load the model and feature engineer
        with open(self.model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(self.feature_engineer_path, 'rb') as f:
            feature_engineer = pickle.load(f)
        
        # Load and prepare test data
        df = self.load_data()
        
        # Take a random sample of data
        test_data = df.sample(n=n_samples, random_state=42)
        
        actual_values = []
        predicted_values = []
        api_predictions = []
        
        print(f"\nTesting with {n_samples} samples from the original dataset...")
        
        for idx, row in test_data.iterrows():
            # Prepare data for API call
            api_data = {
                "DateTime": row['DateTime'].isoformat() if pd.notnull(row['DateTime']) else None,
                "CO(GT)": float(row['CO(GT)']),
                "PT08.S1(CO)": float(row['PT08.S1(CO)']),
                "NMHC(GT)": float(row['NMHC(GT)']),
                "C6H6(GT)": float(row['C6H6(GT)']),
                "PT08.S2(NMHC)": float(row['PT08.S2(NMHC)']),
                "NOx(GT)": float(row['NOx(GT)']),
                "PT08.S3(NOx)": float(row['PT08.S3(NOx)']),
                "NO2(GT)": float(row['NO2(GT)']),
                "PT08.S4(NO2)": float(row['PT08.S4(NO2)']),
                "PT08.S5(O3)": float(row['PT08.S5(O3)']),
                "T": float(row['T']),
                "RH": float(row['RH']),
                "AH": float(row['AH'])
            }
            
            # Make API prediction
            try:
                response = requests.post(f"{API_URL}/predict", json=api_data)
                if response.status_code == 200:
                    api_pred = response.json()['prediction']
                    api_predictions.append(api_pred)
                else:
                    api_predictions.append(np.nan)
            except:
                api_predictions.append(np.nan)
            
            # Direct model prediction (for comparison)
            try:
                df_features = feature_engineer.prepare_single_prediction(api_data)
                X = df_features[feature_engineer.feature_names]
                X_scaled = feature_engineer.scale_features(X)
                model_pred = model.predict(X_scaled)[0]
                predicted_values.append(model_pred)
            except:
                predicted_values.append(np.nan)
            
            actual_values.append(row['CO(GT)'])
        
        # Filter out NaN values
        valid_indices = [i for i, (a, p) in enumerate(zip(actual_values, predicted_values)) 
                        if not (np.isnan(a) or np.isnan(p))]
        
        actual_values_valid = [actual_values[i] for i in valid_indices]
        predicted_values_valid = [predicted_values[i] for i in valid_indices]
        api_predictions_valid = [api_predictions[i] for i in valid_indices if not np.isnan(api_predictions[i])]
        
        # Calculate metrics
        mae = mean_absolute_error(actual_values_valid, predicted_values_valid)
        rmse = np.sqrt(mean_squared_error(actual_values_valid, predicted_values_valid))
        r2 = r2_score(actual_values_valid, predicted_values_valid)
        
        print(f"\nModel Performance Metrics:")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.scatter(actual_values_valid, predicted_values_valid, alpha=0.5)
        plt.plot([min(actual_values_valid), max(actual_values_valid)], 
                [min(actual_values_valid), max(actual_values_valid)], 
                'r--', label='Perfect prediction')
        plt.xlabel('Actual CO(GT) values')
        plt.ylabel('Predicted CO(GT) values')
        plt.title('Actual vs Predicted CO(GT) values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('model_evaluation.png')
        plt.close()
        
        # Create error distribution plot
        errors = np.array(predicted_values_valid) - np.array(actual_values_valid)
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=30, edgecolor='black')
        plt.xlabel('Prediction Error (Predicted - Actual)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Errors')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('error_distribution.png')
        plt.close()
        
        print(f"\nPlots saved as 'model_evaluation.png' and 'error_distribution.png'")
        
        # Compare direct model predictions with API predictions
        if api_predictions_valid:
            api_mae = mean_absolute_error(actual_values_valid[:len(api_predictions_valid)], 
                                        api_predictions_valid)
            print(f"\nAPI prediction MAE: {api_mae:.4f}")
            print(f"Direct model prediction MAE: {mae:.4f}")
        
        # Show some example predictions
        print("\nExample predictions (first 10):")
        print("Actual\tPredicted\tError")
        for i in range(min(10, len(actual_values_valid))):
            error = predicted_values_valid[i] - actual_values_valid[i]
            print(f"{actual_values_valid[i]:.3f}\t{predicted_values_valid[i]:.3f}\t{error:.3f}")

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.evaluate_with_test_data()