from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import pandas as pd
import numpy as np
import json
import os
from typing import Optional

# Create FastAPI app
app = FastAPI(title="Stress Level Prediction System", 
              description="ML-based stress level prediction using various algorithms")

# Simple prediction function for demo (replace with actual model loading later)
def simple_stress_prediction(age, sleep_duration, physical_activity, heart_rate, quality_of_sleep):
    """Simple rule-based stress prediction for demo"""
    stress_score = 5.0  # baseline
    
    # Age factor
    if age > 50:
        stress_score += 0.5
    elif age < 25:
        stress_score += 0.3
    
    # Sleep factor
    if sleep_duration < 6:
        stress_score += 1.5
    elif sleep_duration < 7:
        stress_score += 0.8
    elif sleep_duration > 9:
        stress_score += 0.3
    
    # Physical activity factor
    if physical_activity < 30:
        stress_score += 1.2
    elif physical_activity > 120:
        stress_score -= 0.5
    
    # Heart rate factor
    if heart_rate > 90:
        stress_score += 1.0
    elif heart_rate > 80:
        stress_score += 0.5
    
    # Sleep quality factor
    if quality_of_sleep < 5:
        stress_score += 1.5
    elif quality_of_sleep < 7:
        stress_score += 0.7
    elif quality_of_sleep >= 8:
        stress_score -= 0.5
    
    return max(1.0, min(10.0, stress_score))

# Health recommendations
def get_health_recommendations(stress_level, sleep_duration, physical_activity, heart_rate):
    """Generate health recommendations based on prediction"""
    recommendations = []
    
    if stress_level > 7:
        recommendations.append("🧘‍♀️ Consider stress management techniques like meditation or deep breathing")
        recommendations.append("👨‍⚕️ Consult with a healthcare professional about stress reduction strategies")
        recommendations.append("🏃‍♂️ Regular exercise can significantly help reduce stress levels")
    elif stress_level > 5:
        recommendations.append("💪 Moderate stress detected - try regular exercise and good sleep hygiene")
        recommendations.append("🎵 Consider relaxing activities like music, reading, or nature walks")
    else:
        recommendations.append("✨ Great! Your stress levels appear to be in a healthy range")
        recommendations.append("🎯 Keep maintaining your current healthy lifestyle")
    
    if sleep_duration < 6:
        recommendations.append("😴 Aim for 7-9 hours of sleep per night for better health")
        recommendations.append("📱 Consider limiting screen time before bedtime")
    elif sleep_duration > 9:
        recommendations.append("🤔 Consider if you need this much sleep or if there are underlying issues")
    
    if physical_activity < 30:
        recommendations.append("🚶‍♀️ Try to get at least 30 minutes of physical activity most days")
        recommendations.append("🏃‍♂️ Start with light activities like walking or stretching")
    
    if heart_rate > 100:
        recommendations.append("❤️ Consider consulting a doctor about your elevated resting heart rate")
    elif heart_rate < 60 and physical_activity < 60:
        recommendations.append("💓 Low heart rate detected - ensure this is normal for you")
    
    return recommendations

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main page"""
    try:
        with open("public/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head><title>Stress Level Prediction</title>
        <script src="https://cdn.tailwindcss.com"></script></head>
        <body class="bg-gray-100 p-8">
        <div class="max-w-2xl mx-auto">
        <h1 class="text-3xl font-bold text-center mb-8">🧠 Stress Level Prediction API</h1>
        <div class="bg-white p-6 rounded-lg shadow">
        <p class="mb-4">Welcome to the Stress Level Prediction System!</p>
        <div class="space-y-2">
            <p><a href="/docs" class="text-blue-600 hover:underline">📚 API Documentation</a></p>
            <p><a href="/health" class="text-blue-600 hover:underline">❤️ Health Check</a></p>
            <p><a href="/demo" class="text-blue-600 hover:underline">🎯 Demo Form</a></p>
        </div>
        </div>
        </div>
        </body>
        </html>
        """)

@app.get("/demo", response_class=HTMLResponse)
async def demo_form():
    """Simple demo form"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head><title>Stress Prediction Demo</title>
    <script src="https://cdn.tailwindcss.com"></script></head>
    <body class="bg-gray-100 p-8">
    <div class="max-w-2xl mx-auto">
    <h1 class="text-3xl font-bold text-center mb-8">🧠 Stress Level Prediction Demo</h1>
    <form id="demoForm" class="bg-white p-6 rounded-lg shadow space-y-4">
        <div>
            <label class="block text-sm font-medium mb-2">Age:</label>
            <input type="number" name="age" value="30" class="w-full p-2 border rounded">
        </div>
        <div>
            <label class="block text-sm font-medium mb-2">Sleep Duration (hours):</label>
            <input type="number" name="sleep_duration" value="7" step="0.1" class="w-full p-2 border rounded">
        </div>
        <div>
            <label class="block text-sm font-medium mb-2">Physical Activity (min/day):</label>
            <input type="number" name="physical_activity_level" value="30" class="w-full p-2 border rounded">
        </div>
        <div>
            <label class="block text-sm font-medium mb-2">Heart Rate (BPM):</label>
            <input type="number" name="heart_rate" value="70" class="w-full p-2 border rounded">
        </div>
        <div>
            <label class="block text-sm font-medium mb-2">Sleep Quality (1-10):</label>
            <input type="number" name="quality_of_sleep" value="7" min="1" max="10" class="w-full p-2 border rounded">
        </div>
        <button type="submit" class="w-full bg-blue-600 text-white p-3 rounded hover:bg-blue-700">
            🔮 Predict Stress Level
        </button>
    </form>
    <div id="result" class="mt-6 p-4 bg-white rounded shadow hidden"></div>
    </div>
    <script>
    document.getElementById('demoForm').onsubmit = async (e) => {
        e.preventDefault();
        const formData = new FormData(e.target);
        try {
            const response = await fetch('/predict-simple', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            document.getElementById('result').innerHTML = `
                <h3 class="text-xl font-bold mb-2">Prediction Results</h3>
                <p class="text-2xl text-blue-600 font-bold">Stress Level: ${result.stress_level}/10</p>
                <p class="text-lg">${result.description}</p>
                <div class="mt-4">
                    <h4 class="font-semibold">Recommendations:</h4>
                    <ul class="list-disc pl-5">
                        ${result.recommendations.map(r => `<li>${r}</li>`).join('')}
                    </ul>
                </div>
            `;
            document.getElementById('result').classList.remove('hidden');
        } catch (error) {
            alert('Error: ' + error.message);
        }
    };
    </script>
    </body>
    </html>
    """)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Stress Level Prediction API is running",
        "version": "1.0.0",
        "endpoints": ["/", "/health", "/demo", "/predict-simple", "/docs"]
    }

@app.post("/predict-simple")
async def predict_stress_simple(
    age: int = Form(...),
    sleep_duration: float = Form(...),
    physical_activity_level: int = Form(...),
    heart_rate: int = Form(...),
    quality_of_sleep: int = Form(...)
):
    """Simple stress prediction without complex models"""
    try:
        # Make prediction using simple rules
        stress_level = simple_stress_prediction(
            age, sleep_duration, physical_activity_level, heart_rate, quality_of_sleep
        )
        
        # Generate description
        if stress_level <= 3:
            description = "Low stress - You're managing well!"
        elif stress_level <= 6:
            description = "Moderate stress - Some areas for improvement"
        elif stress_level <= 8:
            description = "High stress - Consider stress management techniques"
        else:
            description = "Very high stress - Recommend professional consultation"
        
        # Generate recommendations
        recommendations = get_health_recommendations(
            stress_level, sleep_duration, physical_activity_level, heart_rate
        )
        
        return {
            "stress_level": round(stress_level, 1),
            "description": description,
            "recommendations": recommendations,
            "input_summary": {
                "age": age,
                "sleep_duration": sleep_duration,
                "physical_activity_level": physical_activity_level,
                "heart_rate": heart_rate,
                "quality_of_sleep": quality_of_sleep
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/predict-simple")
async def predict_stress_simple_get(
    age: int,
    sleep_duration: float,
    physical_activity_level: int,
    heart_rate: int,
    quality_of_sleep: int
):
    """GET endpoint for simple stress prediction"""
    return await predict_stress_simple(
        age, sleep_duration, physical_activity_level, heart_rate, quality_of_sleep
    )

# For Vercel
def handler(request, response):
    return app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
