# src/api/test_improved_client.py

import requests
import json
from datetime import datetime, timedelta
import time

API_URL = "http://localhost:8080"

def test_improved_server():
    """Test the improved server with multiple predictions"""
    
    # Test health endpoint
    try:
        response = requests.get(f"{API_URL}/health")
        print("Health Check Status Code:", response.status_code)
        print("Health Check Response:", json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Health check failed: {e}")
        return
    
    # Test with multiple sequential predictions to build history
    base_values = {
        "CO(GT)": [2.6, 2.8, 2.4, 2.9, 2.7],
        "PT08.S1(CO)": [1360, 1380, 1340, 1400, 1370],
        "NMHC(GT)": [150, 155, 145, 160, 152],
        "C6H6(GT)": [11.9, 12.1, 11.7, 12.3, 12.0],
        "PT08.S2(NMHC)": [1046, 1060, 1035, 1070, 1050],
        "NOx(GT)": [166, 170, 162, 175, 168],
        "PT08.S3(NOx)": [1056, 1070, 1045, 1080, 1060],
        "NO2(GT)": [113, 115, 111, 117, 114],
        "PT08.S4(NO2)": [1692, 1700, 1685, 1710, 1695],
        "PT08.S5(O3)": [1268, 1280, 1260, 1290, 1275],
        "T": [13.6, 14.0, 13.3, 14.2, 13.8],
        "RH": [48.9, 49.5, 48.2, 50.1, 49.0],
        "AH": [0.7578, 0.76, 0.75, 0.77, 0.758]
    }
    
    print("\nTesting with sequential predictions to build history:")
    for i in range(5):
        # Create data point with timestamp
        data = {
            "DateTime": (datetime.now() - timedelta(hours=4-i)).isoformat()
        }
        
        # Add sensor values
        for key, values in base_values.items():
            data[key] = values[i]
        
        try:
            response = requests.post(f"{API_URL}/predict", json=data)
            print(f"\nPrediction {i+1}:")
            print(f"Input NO2(GT): {data['NO2(GT)']}")
            result = response.json()
            print(f"Predicted NO2(GT): {result.get('prediction', 'ERROR')}")
            print(f"Status: {result.get('status', 'ERROR')}")
            print(f"Historical data used: {result.get('historical_data_used', False)}")
            
            time.sleep(0.5)  # Small delay between requests
            
        except Exception as e:
            print(f"Prediction {i+1} failed: {e}")

if __name__ == "__main__":
    test_improved_server()