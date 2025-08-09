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

app = FastAPI(title="Stress Level Prediction System", 
              description="ML-based stress level prediction using various algorithms")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

class StressPredictionSystem:
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.scaler = None
        self.selected_features = None
        self.model_results = {}
        self.load_system()
    
    def load_system(self):
        """Load all models and preprocessors"""
        try:
            # Load preprocessors
            self.scaler = joblib.load('models/scaler.pkl')
            self.selected_features = joblib.load('models/selected_features.pkl')
            self.encoders['gender'] = joblib.load('models/gender_encoder.pkl')
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
                except Exception as e:
                    print(f"Error loading {model_file}: {e}")
            
            print(f"Loaded {len(self.models)} models successfully")
            
        except Exception as e:
            print(f"Error loading system: {e}")
            self.models = {}
    
    def preprocess_input(self, data):
        """Preprocess input data for prediction"""
        # Create DataFrame
        df = pd.DataFrame([data])
        
        # Feature engineering (same as training)
        df[['Systolic_BP', 'Diastolic_BP']] = df['Blood_Pressure'].str.split('/', expand=True).astype(int)
        
        df['BMI_Risk_Score'] = df['BMI_Category'].map({
            'Normal': 0, 'Normal Weight': 0, 'Overweight': 1, 'Obese': 2
        })
        
        df['Sleep_Quality_Score'] = df['Quality_of_Sleep'] * df['Sleep_Duration']
        df['Activity_Sleep_Ratio'] = df['Physical_Activity_Level'] / df['Sleep_Duration']
        df['Steps_per_Hour_Awake'] = df['Daily_Steps'] / (24 - df['Sleep_Duration'])
        df['BP_Risk'] = (df['Systolic_BP'] > 130) | (df['Diastolic_BP'] > 80)
        df['Heart_Rate_Category'] = pd.cut(df['Heart_Rate'], 
                                         bins=[0, 60, 100, 200], 
                                         labels=[0, 1, 2])
        
        df['Has_Sleep_Disorder'] = (df['Sleep_Disorder'] != 'None').astype(int)
        df['Sleep_Disorder_Type'] = df['Sleep_Disorder'].map({
            'None': 0, 'Insomnia': 1, 'Sleep Apnea': 2
        })
        
        df['Age_Group'] = pd.cut(df['Age'], 
                               bins=[0, 30, 40, 50, 100], 
                               labels=[0, 1, 2, 3])
        
        # Encode categorical variables
        df['Gender_Encoded'] = self.encoders['gender'].transform(df['Gender'])
        df['Occupation_Encoded'] = self.encoders['occupation'].transform(df['Occupation'])
        
        # Select features and scale
        X = df[self.selected_features]
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def predict_stress(self, input_data):
        """Make predictions using all loaded models"""
        predictions = {}
        
        try:
            # Preprocess input
            X_scaled = self.preprocess_input(input_data)
            
            # Get predictions from all models
            for model_name, model in self.models.items():
                try:
                    if 'regression' in model_name:
                        pred = float(model.predict(X_scaled)[0])
                        predictions[f'{model_name}_prediction'] = round(pred, 2)
                    elif 'classification' in model_name:
                        pred = int(model.predict(X_scaled)[0])
                        predictions[f'{model_name}_prediction'] = pred
                        
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(X_scaled)[0]
                            predictions[f'{model_name}_probabilities'] = [round(p, 3) for p in proba]
                    
                except Exception as e:
                    print(f"Error with model {model_name}: {e}")
                    predictions[f'{model_name}_error'] = str(e)
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            predictions['preprocessing_error'] = str(e)
        
        return predictions

# Initialize the prediction system
predictor = StressPredictionSystem()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with prediction form"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "model_count": len(predictor.models),
        "available_models": list(predictor.models.keys())
    })

@app.get("/models", response_class=HTMLResponse)
async def models_page(request: Request):
    """Models comparison page"""
    return templates.TemplateResponse("models.html", {
        "request": request,
        "model_results": predictor.model_results,
        "models": list(predictor.models.keys())
    })

@app.post("/predict")
async def predict_stress(
    gender: str = Form(...),
    age: int = Form(...),
    occupation: str = Form(...),
    sleep_duration: float = Form(...),
    quality_of_sleep: int = Form(...),
    physical_activity_level: int = Form(...),
    heart_rate: int = Form(...),
    daily_steps: int = Form(...),
    blood_pressure: str = Form(...),
    bmi_category: str = Form(...),
    sleep_disorder: str = Form(...)
):
    """Predict stress level using all available models"""
    
    # Prepare input data
    input_data = {
        'Gender': gender,
        'Age': age,
        'Occupation': occupation,
        'Sleep_Duration': sleep_duration,
        'Quality_of_Sleep': quality_of_sleep,
        'Physical_Activity_Level': physical_activity_level,
        'Heart_Rate': heart_rate,
        'Daily_Steps': daily_steps,
        'Blood_Pressure': blood_pressure,
        'BMI_Category': bmi_category,
        'Sleep_Disorder': sleep_disorder
    }
    
    try:
        # Get predictions
        predictions = predictor.predict_stress(input_data)
        
        # Organize predictions by type
        regression_predictions = {}
        classification_predictions = {}
        
        for key, value in predictions.items():
            if 'regression' in key:
                model_name = key.replace('_regression_prediction', '').replace('_', ' ').title()
                regression_predictions[model_name] = value
            elif 'classification' in key and 'probabilities' not in key:
                model_name = key.replace('_classification_prediction', '').replace('_', ' ').title()
                classification_predictions[model_name] = value
        
        # Calculate ensemble predictions
        if regression_predictions:
            ensemble_regression = round(np.mean(list(regression_predictions.values())), 2)
        else:
            ensemble_regression = None
            
        if classification_predictions:
            ensemble_classification = int(np.round(np.mean(list(classification_predictions.values()))))
        else:
            ensemble_classification = None
        
        # Determine stress level description
        def get_stress_description(level):
            if level <= 3:
                return {"level": "Low", "color": "green", "description": "You appear to have low stress levels. Keep up the good work!"}
            elif level <= 6:
                return {"level": "Moderate", "color": "yellow", "description": "You have moderate stress levels. Consider stress management techniques."}
            else:
                return {"level": "High", "color": "red", "description": "You have high stress levels. Consider consulting a healthcare professional."}
        
        result = {
            "success": True,
            "input_data": input_data,
            "regression_predictions": regression_predictions,
            "classification_predictions": classification_predictions,
            "ensemble_regression": ensemble_regression,
            "ensemble_classification": ensemble_classification,
            "stress_analysis": get_stress_description(ensemble_regression or ensemble_classification or 5),
            "model_count": len(predictor.models)
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/api/model-results")
async def get_model_results():
    """Get model training and testing results"""
    return JSONResponse(content=predictor.model_results)

@app.get("/api/available-models")
async def get_available_models():
    """Get list of available models"""
    return JSONResponse(content={
        "models": list(predictor.models.keys()),
        "count": len(predictor.models)
    })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(content={
        "status": "healthy",
        "models_loaded": len(predictor.models),
        "system_ready": len(predictor.models) > 0
    })

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)