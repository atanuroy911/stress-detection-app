from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
import numpy as np
import json
import os
from typing import Optional
import uvicorn

# Create FastAPI app
app = FastAPI(title="Stress Level Prediction System", 
              description="ML-based stress level prediction using various algorithms")

# Configure for Vercel deployment
try:
    # Mount static files
    if os.path.exists("static"):
        app.mount("/static", StaticFiles(directory="static"), name="static")
    
    # Templates
    if os.path.exists("templates"):
        templates = Jinja2Templates(directory="templates")
    else:
        templates = None
except Exception as e:
    print(f"Warning: Static files or templates not found: {e}")
    templates = None

class StressPredictionSystem:
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.scaler = None
        self.selected_features = None
        self.model_results = {}
        self.is_loaded = False
        self.load_system()
    
    def load_system(self):
        """Load all models and preprocessors"""
        try:
            models_dir = 'models'
            if not os.path.exists(models_dir):
                print("Warning: Models directory not found")
                return
                
            # Load preprocessors
            if os.path.exists('models/scaler.pkl'):
                self.scaler = joblib.load('models/scaler.pkl')
            if os.path.exists('models/selected_features.pkl'):
                self.selected_features = joblib.load('models/selected_features.pkl')
            if os.path.exists('models/gender_encoder.pkl'):
                self.encoders['gender'] = joblib.load('models/gender_encoder.pkl')
            if os.path.exists('models/occupation_encoder.pkl'):
                self.encoders['occupation'] = joblib.load('models/occupation_encoder.pkl')
            
            # Load model results
            if os.path.exists('models/results.json'):
                with open('models/results.json', 'r') as f:
                    self.model_results = json.load(f)
            
            # Load models
            model_files = [f for f in os.listdir('models') if f.endswith('.pkl') and 'encoder' not in f and 'scaler' not in f and 'features' not in f]
            
            for model_file in model_files:
                model_name = model_file.replace('.pkl', '')
                try:
                    self.models[model_name] = joblib.load(f'models/{model_file}')
                    print(f"Loaded model: {model_name}")
                except Exception as e:
                    print(f"Failed to load model {model_name}: {e}")
            
            self.is_loaded = len(self.models) > 0
            print(f"Loaded {len(self.models)} models successfully")
            
        except Exception as e:
            print(f"Error loading system: {e}")
            self.is_loaded = False
    
    def preprocess_input(self, gender, age, occupation, sleep_duration, quality_of_sleep, 
                        physical_activity_level, bmi_category, blood_pressure, 
                        heart_rate, daily_steps, sleep_disorder):
        """Preprocess input data for prediction"""
        
        if not self.is_loaded:
            raise HTTPException(status_code=500, detail="Models not loaded properly")
        
        # Create input data
        input_data = {
            'Gender': gender,
            'Age': float(age),
            'Occupation': occupation,
            'Sleep Duration': float(sleep_duration),
            'Quality of Sleep': float(quality_of_sleep),
            'Physical Activity Level': float(physical_activity_level),
            'BMI Category': bmi_category,
            'Blood Pressure': blood_pressure,
            'Heart Rate': float(heart_rate),
            'Daily Steps': float(daily_steps),
            'Sleep Disorder': sleep_disorder
        }
        
        df = pd.DataFrame([input_data])
        
        # Handle categorical encoding
        if 'gender' in self.encoders:
            df['Gender'] = self.encoders['gender'].transform([gender])[0]
        if 'occupation' in self.encoders:
            df['Occupation'] = self.encoders['occupation'].transform([occupation])[0]
        
        # Handle BMI Category encoding manually (assuming numeric conversion)
        bmi_mapping = {'Underweight': 1, 'Normal': 2, 'Normal Weight': 2, 'Overweight': 3, 'Obese': 4}
        df['BMI Category'] = bmi_mapping.get(bmi_category, 2)
        
        # Handle Blood Pressure (extract systolic)
        if '/' in str(blood_pressure):
            systolic = int(str(blood_pressure).split('/')[0])
        else:
            systolic = int(blood_pressure)
        df['Blood Pressure'] = systolic
        
        # Handle Sleep Disorder encoding
        sleep_disorder_mapping = {'None': 0, 'Sleep Apnea': 1, 'Insomnia': 2}
        df['Sleep Disorder'] = sleep_disorder_mapping.get(sleep_disorder, 0)
        
        # Select features if available
        if self.selected_features is not None:
            df = df[self.selected_features]
        
        # Scale features if scaler is available
        if self.scaler is not None:
            df_scaled = self.scaler.transform(df)
        else:
            df_scaled = df.values
        
        return df_scaled
    
    def predict(self, processed_data):
        """Make predictions using all loaded models"""
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                prediction = model.predict(processed_data)[0]
                predictions[model_name] = float(prediction)
            except Exception as e:
                print(f"Error predicting with {model_name}: {e}")
                predictions[model_name] = None
        
        return predictions

# Initialize prediction system
prediction_system = StressPredictionSystem()

# Health recommendations
def get_health_recommendations(stress_level, sleep_duration, physical_activity, heart_rate):
    """Generate health recommendations based on prediction"""
    recommendations = []
    
    if stress_level > 7:
        recommendations.append("Consider stress management techniques like meditation or deep breathing")
        recommendations.append("Consult with a healthcare professional about stress reduction strategies")
    elif stress_level > 5:
        recommendations.append("Moderate stress detected - try regular exercise and good sleep hygiene")
    else:
        recommendations.append("Great! Your stress levels appear to be in a healthy range")
    
    if sleep_duration < 6:
        recommendations.append("Aim for 7-9 hours of sleep per night for better health")
    elif sleep_duration > 9:
        recommendations.append("Consider if you need this much sleep or if there are underlying issues")
    
    if physical_activity < 30:
        recommendations.append("Try to get at least 30 minutes of physical activity most days")
    
    if heart_rate > 100:
        recommendations.append("Consider consulting a doctor about your elevated resting heart rate")
    elif heart_rate < 60:
        recommendations.append("Low heart rate detected - ensure this is normal for you")
    
    return recommendations

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    if templates:
        return templates.TemplateResponse("index.html", {"request": request})
    else:
        return HTMLResponse("""
        <html>
        <head><title>Stress Level Prediction</title></head>
        <body>
        <h1>Stress Level Prediction API</h1>
        <p>Welcome to the Stress Level Prediction System!</p>
        <p>Use the API endpoints:</p>
        <ul>
            <li><a href="/docs">API Documentation</a></li>
            <li><a href="/health">Health Check</a></li>
            <li><a href="/models">Available Models</a></li>
        </ul>
        </body>
        </html>
        """)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": prediction_system.is_loaded,
        "available_models": list(prediction_system.models.keys()) if prediction_system.is_loaded else []
    }

@app.get("/models")
async def get_models():
    """Get available models and their performance"""
    if not prediction_system.is_loaded:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    return {
        "available_models": list(prediction_system.models.keys()),
        "model_results": prediction_system.model_results,
        "total_models": len(prediction_system.models)
    }

@app.post("/predict")
async def predict_stress(
    gender: str = Form(...),
    age: int = Form(...),
    occupation: str = Form(...),
    sleep_duration: float = Form(...),
    quality_of_sleep: int = Form(...),
    physical_activity_level: int = Form(...),
    bmi_category: str = Form(...),
    blood_pressure: str = Form(...),
    heart_rate: int = Form(...),
    daily_steps: int = Form(...),
    sleep_disorder: str = Form(...)
):
    """Predict stress level"""
    try:
        # Preprocess input
        processed_data = prediction_system.preprocess_input(
            gender, age, occupation, sleep_duration, quality_of_sleep,
            physical_activity_level, bmi_category, blood_pressure,
            heart_rate, daily_steps, sleep_disorder
        )
        
        # Make predictions
        predictions = prediction_system.predict(processed_data)
        
        # Calculate ensemble prediction (average of valid predictions)
        valid_predictions = [p for p in predictions.values() if p is not None]
        ensemble_prediction = np.mean(valid_predictions) if valid_predictions else 5.0
        
        # Generate recommendations
        recommendations = get_health_recommendations(
            ensemble_prediction, sleep_duration, physical_activity_level, heart_rate
        )
        
        return {
            "ensemble_prediction": round(ensemble_prediction, 2),
            "individual_predictions": predictions,
            "recommendations": recommendations,
            "input_summary": {
                "age": age,
                "sleep_duration": sleep_duration,
                "physical_activity_level": physical_activity_level,
                "heart_rate": heart_rate
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/predict")
async def predict_stress_get(
    gender: str,
    age: int,
    occupation: str,
    sleep_duration: float,
    quality_of_sleep: int,
    physical_activity_level: int,
    bmi_category: str,
    blood_pressure: str,
    heart_rate: int,
    daily_steps: int,
    sleep_disorder: str
):
    """GET endpoint for stress prediction"""
    return await predict_stress(
        gender, age, occupation, sleep_duration, quality_of_sleep,
        physical_activity_level, bmi_category, blood_pressure,
        heart_rate, daily_steps, sleep_disorder
    )

# Vercel handler
app_handler = app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
