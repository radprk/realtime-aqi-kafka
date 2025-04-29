# Enhanced Real-Time Air Quality Monitoring System

## Overview

This repository contains the implementation of the "Enhanced Real-Time Air Quality Monitoring System" for the course *94879-A4: Fundamentals of Operationalizing AI* (Spring 2025). The project builds on individual Kafka-based air quality prediction models, extending them into a production-ready system with a comprehensive MLOps pipeline. We use the UCI Air Quality dataset to stream, process, and predict pollutant levels like CO(GT) in real time, achieving at least a 10% improvement over individual assignment baselines.

### Team Members
- Devan Rajendran
- Dharmesh Agase
- Drisya Antose
- Radha Parikh
- Shrenya Mathur

## Project Architecture

The system architecture includes the following components:
- **Kafka Producer**: Streams the UCI Air Quality dataset into the `air-quality-data` topic using batch streaming to simulate real-time sensor data.
- **Kafka Consumer**: Retrieves batched data, preprocesses it (e.g., handling missing values, creating a datetime index), and sends it to a prediction API.
- **Machine Learning Models**: Models like XGBoost, Random Forest, and LSTM were trained and tracked using MLflow. The best model (XGBoost, MAE: 0.18) was selected, achieving a 92.8% improvement over the baseline (MAE 2.50).
- **Docker API**: The prediction model is containerized using Docker and exposed via a FastAPI-based API on port 8080.
- **Output**: Predictions are saved to `data/processed/processed_air_quality_data.csv` for review.

## Prerequisites

- Python 3.8+
- Apache Kafka (with Zookeeper)
- Docker
- MLflow
- FastAPI
- Required Python packages (listed in `requirements.txt`)

## Setup Instructions

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/radprk/realtime-aqi-kafka.git
   cd realtime-aqi-kafka
   ```

2. **Install Dependencies**  
   Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Kafka**  
   - Download and install Apache Kafka.
   - Start Zookeeper:
     ```bash
     ./bin/zookeeper-server-start.sh config/zookeeper.properties
     ```
   - Start Kafka server:
     ```bash
     ./bin/kafka-server-start.sh config/server.properties
     ```
   - Create the Kafka topic:
     ```bash
     ./bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic air-quality-data
     ```

4. **Set Up MLflow**  
   - Start the MLflow tracking server:
     ```bash
     mlflow ui --backend-store-uri sqlite:///mlflow.db
     ```
   - Ensure your trained model (`best_model.pkl`) and `feature_names.json` are in the appropriate directory.

5. **Build and Run the Docker Container**  
   - Build the Docker image for the prediction API:
     ```bash
     docker build -t aqi-prediction-api .
     ```
   - Run the container, mapping port 8080:
     ```bash
     docker run --name airquality-container -p 8080:8080 airquality-api
     ```


## Usage

1. **Run the Kafka Producer**  
   Stream the UCI Air Quality dataset into the Kafka topic:
   ```bash
   python producer.py
   ```

2. **Run the Kafka Consumer**  
   Consume data from the topic, preprocess it, and send it to the Docker-hosted API for predictions:
   ```bash
   python consumer.py
   ```

3. **Check Predictions**  
   Predictions are saved to `data/processed/processed_air_quality_data.csv`. 


## Model Performance

We experimented with several models, tracked via MLflow, aiming for at least a 10% improvement over individual baselines:
- **XGBoost**: MAE 0.18 (best model, used for deployment, 92.8% improvement over baseline)
- **Random Forest**: MAE 0.19
- **LSTM**: MAE 0.22–1.57 (high variability)
- **Ensemble**: MAE 0.21
- **Linear Regression**: MAE 12.47 (baseline comparison)

The baseline MAE was 2.50, and our best model (XGBoost) far exceeded the 10% improvement goal.

## Challenges and Solutions

- **Kafka Integration**: Resolved missing Zookeeper scripts by adjusting the startup process and implemented real-time feature engineering in the consumer to handle raw data.
- **Docker Issues**: Fixed port conflicts and missing files in the container by using dynamic port mapping and ensuring all artifacts were included.
- **API Integration**: Addressed feature mismatches by adding real-time feature engineering and handled data type inconsistencies with explicit conversions.

## Future Improvements

- Enhance the ensemble model by refining weighting strategies for better performance.
- Stabilize the LSTM model through improved hyperparameter tuning.
- Scale the system for larger datasets using multiple Kafka partitions.
- Kubernetes deployment of containers to a cloud environment. 
- Include Evidently monitoring to show advanced features beyond a basic data drift dashboard.

## Acknowledgments

- UCI Machine Learning Repository for providing the Air Quality dataset.
- Prof. Anand Rao and the teaching staff of 94879-A4 for guidance and resources.

---

For any questions, please contact the team via Canvas or Piazza.
