import os
import pandas as pd
import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ✅ Load trained ML models
bacteria_model = joblib.load("models/bacteria_model.pkl")
do_model = joblib.load("models/do_model.pkl")
metal_model = joblib.load("models/metal_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ✅ CSV file for storing sensor data
CSV_FILE = "sensor_data.csv"

# ✅ Initialize FastAPI
app = FastAPI(title="Water Quality Monitoring API")

# ✅ Enable CORS (for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Define Input Schema
class WaterQualityInput(BaseModel):
    temperature: float
    ph: float
    tds: float
    turbidity: float

# ✅ Function to read sensor data from CSV
def get_sensor_data():
    try:
        df = pd.read_csv(CSV_FILE)
        if df.empty:
            return None
        latest_data = df.iloc[-1]  # Get the most recent sensor values
        return {
            "temperature": float(latest_data["temperature"]),
            "ph": float(latest_data["ph"]),
            "tds": float(latest_data["tds"]),
            "turbidity": float(latest_data["turbidity"]),
        }
    except Exception as e:
        print(f"Error reading sensor data: {e}")
        return None

# ✅ Function to append new sensor data to CSV
def update_sensor_data(sensor_data):
    try:
        df = pd.DataFrame([sensor_data])  # Convert dictionary to DataFrame
        if not os.path.exists(CSV_FILE):
            df.to_csv(CSV_FILE, index=False)  # Create file if not exists
        else:
            df.to_csv(CSV_FILE, mode='a', header=False, index=False)  # Append new data
    except Exception as e:
        print(f"Error updating sensor data: {e}")

@app.head("/")
def head():
    return {}  # No response body, just headers

@app.get("/")
def home():
    return {"message": "Water Quality API is running"}

# ✅ API Endpoint to Submit New Sensor Data
@app.post("/submit_data")
def submit_data(sensor_data: WaterQualityInput):
    new_data = {
        "temperature": sensor_data.temperature,
        "ph": sensor_data.ph,
        "tds": sensor_data.tds,
        "turbidity": sensor_data.turbidity,
    }
    update_sensor_data(new_data)
    return {"message": "Sensor data updated successfully"}

# ✅ API Endpoint for Predictions
@app.get("/predict")
def predict_water_quality():
    sensor_data = get_sensor_data()
    if sensor_data is None:
        return {"error": "No sensor data available"}

    input_data = np.array([[sensor_data["temperature"], sensor_data["ph"], sensor_data["tds"], sensor_data["turbidity"]]])
    input_scaled = scaler.transform(input_data)

    pred_do = do_model.predict(input_scaled)[0]
    pred_metal = metal_model.predict(input_scaled)[0]
    pred_bacteria = bacteria_model.predict(input_scaled)[0]

    return {
        "temperature": sensor_data["temperature"],
        "ph": sensor_data["ph"],
        "tds": sensor_data["tds"],
        "turbidity": sensor_data["turbidity"],
        "Dissolved Oxygen (DO)": f"{pred_do:.2f} mg/L",
        "Heavy Metal Concentration": f"{pred_metal:.4f} mg/L",
        "Bacterial Contamination": "Contaminated" if pred_bacteria == 1 else "Safe"
    }

# ✅ Start Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
