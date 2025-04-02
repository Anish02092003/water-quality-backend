import os
import requests
import joblib
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ✅ Load environment variables
load_dotenv()
BLYNK_AUTH_TOKEN = os.getenv("JyZsPsdPWqRMFeG9q90YK5DOlNU5dXp6")
BLYNK_URL = "https://blynk.cloud/external/api?token=JyZsPsdPWqRMFeG9q90YK5DOlNU5dXp6"


# ✅ Load trained ML models
bacteria_model = joblib.load("models/bacteria_model.pkl")
do_model = joblib.load("models/do_model.pkl")
metal_model = joblib.load("models/metal_model.pkl")
scaler = joblib.load("models/scaler.pkl")

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

# ✅ Function to Get Real-Time Sensor Data from Blynk
def get_sensor_data():
    try:
        response = requests.get(f"{BLYNK_URL}get?token={BLYNK_AUTH_TOKEN}&V1&V2&V3&V4")
        data = response.json()
        return {
            "temperature": float(data[0]),  
            "ph": float(data[1]),
            "tds": float(data[2]),
            "turbidity": float(data[3]),
        }
    except Exception as e:
        print(f"Error fetching data from Blynk: {e}")
        return None

# ✅ Function to Send Predictions to Blynk
def send_to_blynk(bacteria, do, metal):
    payload = {
        "V5": do,  # Display DO on Blynk Virtual Pin V5
        "V6": metal,  # Display Metal Concentration on V6
        "V7": bacteria,  # Display Bacterial Contamination on V7
    }
    requests.get(f"{BLYNK_URL}update?token={BLYNK_AUTH_TOKEN}", params=payload)

# ✅ API Endpoint for Predictions
@app.get("/predict")
def predict_water_quality():
    sensor_data = get_sensor_data()
    if sensor_data is None:
        return {"error": "Failed to fetch sensor data"}

    input_data = np.array([[sensor_data["temperature"], sensor_data["ph"], sensor_data["tds"], sensor_data["turbidity"]]])
    input_scaled = scaler.transform(input_data)

    pred_do = do_model.predict(input_scaled)[0]
    pred_metal = metal_model.predict(input_scaled)[0]
    pred_bacteria = bacteria_model.predict(input_scaled)[0]

    # ✅ Send predictions to Blynk
    send_to_blynk(pred_bacteria, pred_do, pred_metal)

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

