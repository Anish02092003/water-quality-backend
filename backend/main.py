from fastapi import FastAPI
import joblib
import numpy as np
import os
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Load trained ML models
bacteria_model = joblib.load("models/bacteria_model.pkl")
do_model = joblib.load("models/do_model.pkl")
metal_model = joblib.load("models/metal_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Initialize FastAPI
app = FastAPI(title="Water Quality Prediction API")

# Define Input Schema
class WaterQualityInput(BaseModel):
    temperature: float
    ph: float
    tds: float
    turbidity: float

@app.post("/predict")
def predict_water_quality(data: WaterQualityInput):
    input_data = np.array([[data.temperature, data.ph, data.tds, data.turbidity]])
    input_scaled = scaler.transform(input_data)

    pred_do = do_model.predict(input_scaled)[0]
    pred_metal = metal_model.predict(input_scaled)[0]
    pred_bacteria = bacteria_model.predict(input_scaled)[0]

    return {
        "Dissolved Oxygen (DO)": f"{pred_do:.2f} mg/L",
        "Heavy Metal Concentration": f"{pred_metal:.4f} mg/L",
        "Bacterial Contamination": "Contaminated" if pred_bacteria == 1 else "Safe"
    }
app = FastAPI()
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.head("/")
def head():
    return {}  # No response body, just headers
@app.get("/")
def home():
    return {}
    if __name__ == "__main__":
