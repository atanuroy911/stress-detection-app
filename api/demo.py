from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import json
import os
from typing import Optional

# Create FastAPI app
app = FastAPI(title="Stress Level Prediction System", 
              description="ML-based stress level prediction using rule-based algorithms")

# Simple prediction function (no ML libraries needed)
def simple_stress_prediction(age, sleep_duration, physical_activity, heart_rate, quality_of_sleep):
    """Rule-based stress prediction algorithm"""
    stress_score = 5.0  # baseline
    
    # Age factor (older age tends to have different stress patterns)
    if age > 50:
        stress_score += 0.5
    elif age < 25:
        stress_score += 0.3
    
    # Sleep factor (most important)
    if sleep_duration < 6:
        stress_score += 2.0  # Severe sleep deprivation
    elif sleep_duration < 7:
        stress_score += 1.2  # Mild sleep deprivation
    elif sleep_duration > 9:
        stress_score += 0.4  # Excessive sleep may indicate issues
    else:
        stress_score -= 0.3  # Optimal sleep
    
    # Physical activity factor
    if physical_activity < 20:
        stress_score += 1.5  # Sedentary lifestyle
    elif physical_activity < 30:
        stress_score += 0.8  # Low activity
    elif physical_activity > 120:
        stress_score += 0.3  # Excessive exercise can be stressful
    else:
        stress_score -= 0.5  # Good activity level
    
    # Heart rate factor (resting heart rate indicator)
    if heart_rate > 100:
        stress_score += 1.5  # Very high resting HR
    elif heart_rate > 85:
        stress_score += 1.0  # High resting HR
    elif heart_rate > 75:
        stress_score += 0.5  # Elevated HR
    elif heart_rate < 50 and physical_activity < 60:
        stress_score += 0.3  # Unusually low (if not athletic)
    else:
        stress_score -= 0.2  # Normal range
    
    # Sleep quality factor
    if quality_of_sleep <= 3:
        stress_score += 2.0  # Very poor sleep quality
    elif quality_of_sleep <= 5:
        stress_score += 1.2  # Poor sleep quality
    elif quality_of_sleep <= 7:
        stress_score += 0.3  # Fair sleep quality
    else:
        stress_score -= 0.5  # Good sleep quality
    
    # Ensure score is within valid range
    return max(1.0, min(10.0, round(stress_score, 1)))

def get_stress_category(score):
    """Get stress category and description"""
    if score <= 3:
        return "Low", "You're managing stress very well! Keep up the good work."
    elif score <= 5:
        return "Low-Moderate", "Your stress levels are mostly healthy with room for minor improvements."
    elif score <= 7:
        return "Moderate", "You're experiencing moderate stress. Consider implementing stress management techniques."
    elif score <= 8.5:
        return "High", "High stress levels detected. It's important to focus on stress reduction strategies."
    else:
        return "Very High", "Very high stress levels. Consider consulting with a healthcare professional."

def get_health_recommendations(stress_level, sleep_duration, physical_activity, heart_rate, quality_of_sleep):
    """Generate personalized health recommendations"""
    recommendations = []
    
    # Stress-based recommendations
    if stress_level > 8:
        recommendations.extend([
            "🧘‍♀️ Practice daily meditation or deep breathing exercises (10-15 minutes)",
            "👨‍⚕️ Consider consulting with a healthcare professional or counselor",
            "🏃‍♂️ Engage in regular moderate exercise to reduce stress hormones",
            "🛀 Try relaxation techniques like warm baths or progressive muscle relaxation"
        ])
    elif stress_level > 6:
        recommendations.extend([
            "💪 Incorporate stress-reducing activities like yoga or tai chi",
            "🎵 Listen to calming music or practice mindfulness",
            "🌿 Spend time in nature or green spaces when possible",
            "📚 Consider stress management books or apps"
        ])
    else:
        recommendations.extend([
            "✨ Great job managing your stress levels!",
            "🎯 Continue your current healthy lifestyle habits",
            "🔄 Maintain consistency in your wellness routine"
        ])
    
    # Sleep-based recommendations
    if sleep_duration < 6:
        recommendations.extend([
            "😴 Aim for 7-9 hours of sleep per night for optimal health",
            "📱 Limit screen time 1-2 hours before bedtime",
            "🌙 Create a consistent bedtime routine",
            "☕ Avoid caffeine 6+ hours before sleep"
        ])
    elif sleep_duration > 9:
        recommendations.append("🤔 Monitor if excessive sleep is due to underlying health issues")
    
    if quality_of_sleep <= 5:
        recommendations.extend([
            "🛏️ Optimize your sleep environment (cool, dark, quiet)",
            "💤 Consider a sleep study if quality doesn't improve",
            "🧘‍♀️ Try relaxation techniques before bed"
        ])
    
    # Activity-based recommendations  
    if physical_activity < 30:
        recommendations.extend([
            "🚶‍♀️ Start with 30 minutes of walking daily",
            "🏃‍♂️ Gradually increase activity with activities you enjoy",
            "🎯 Set realistic fitness goals and track progress"
        ])
    elif physical_activity > 120:
        recommendations.append("⚖️ Ensure adequate rest days to prevent overtraining stress")
    
    # Heart rate recommendations
    if heart_rate > 100:
        recommendations.extend([
            "❤️ Monitor heart rate regularly and consult a doctor",
            "🧘‍♀️ Practice heart rate variability breathing exercises"
        ])
    elif heart_rate > 85:
        recommendations.append("💓 Consider cardiovascular fitness activities")
    
    return recommendations[:8]  # Limit to top 8 recommendations

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main page"""
    try:
        with open("public/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Stress Level Prediction System</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
        </head>
        <body class="bg-gradient-to-br from-blue-50 to-indigo-100 min-h-screen">
            <div class="container mx-auto px-4 py-8">
                <h1 class="text-4xl font-bold text-center text-indigo-800 mb-4">
                    🧠 Stress Level Prediction System
                </h1>
                <p class="text-center text-gray-600 mb-8">AI-Powered Health Assessment Tool</p>
                
                <div class="max-w-2xl mx-auto bg-white rounded-xl shadow-lg p-8">
                    <h2 class="text-2xl font-semibold text-gray-800 mb-6">Quick Health Assessment</h2>
                    
                    <form id="assessmentForm" class="space-y-4">
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                                <label class="block text-sm font-medium text-gray-700 mb-2">Age</label>
                                <input type="number" name="age" value="30" min="18" max="100" 
                                       class="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-indigo-500">
                            </div>
                            
                            <div>
                                <label class="block text-sm font-medium text-gray-700 mb-2">Sleep Duration (hours)</label>
                                <input type="number" name="sleep_duration" value="7" min="4" max="12" step="0.1"
                                       class="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-indigo-500">
                            </div>
                            
                            <div>
                                <label class="block text-sm font-medium text-gray-700 mb-2">Physical Activity (min/day)</label>
                                <input type="number" name="physical_activity_level" value="30" min="0" max="300"
                                       class="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-indigo-500">
                            </div>
                            
                            <div>
                                <label class="block text-sm font-medium text-gray-700 mb-2">Heart Rate (BPM)</label>
                                <input type="number" name="heart_rate" value="70" min="40" max="200"
                                       class="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-indigo-500">
                            </div>
                            
                            <div class="md:col-span-2">
                                <label class="block text-sm font-medium text-gray-700 mb-2">Sleep Quality (1-10 scale)</label>
                                <select name="quality_of_sleep" class="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-indigo-500">
                                    <option value="1">1 - Very Poor</option>
                                    <option value="2">2 - Poor</option>
                                    <option value="3">3 - Below Average</option>
                                    <option value="4">4 - Fair</option>
                                    <option value="5">5 - Average</option>
                                    <option value="6">6 - Above Average</option>
                                    <option value="7" selected>7 - Good</option>
                                    <option value="8">8 - Very Good</option>
                                    <option value="9">9 - Excellent</option>
                                    <option value="10">10 - Perfect</option>
                                </select>
                            </div>
                        </div>
                        
                        <button type="submit" class="w-full bg-indigo-600 text-white py-3 px-6 rounded-md hover:bg-indigo-700 transition duration-200 font-semibold text-lg">
                            🔮 Analyze My Stress Level
                        </button>
                    </form>
                    
                    <div id="results" class="mt-8 p-6 bg-gray-50 rounded-lg hidden">
                        <div id="resultsContent"></div>
                    </div>
                </div>
                
                <div class="max-w-2xl mx-auto mt-8 text-center text-sm text-gray-500">
                    <p>⚠️ This tool is for educational purposes only and should not replace professional medical advice.</p>
                </div>
            </div>

            <script>
                document.getElementById('assessmentForm').onsubmit = async (e) => {{
                    e.preventDefault();
                    const formData = new FormData(e.target);
                    
                    try {{
                        const response = await fetch('/predict', {{
                            method: 'POST',
                            body: formData
                        }});
                        
                        const result = await response.json();
                        displayResults(result);
                    }} catch (error) {{
                        console.error('Error:', error);
                        alert('Analysis failed. Please try again.');
                    }}
                }};
                
                function displayResults(result) {{
                    const resultsDiv = document.getElementById('results');
                    const resultsContent = document.getElementById('resultsContent');
                    
                    const stressColor = result.stress_level > 7 ? 'text-red-600' : 
                                       result.stress_level > 5 ? 'text-yellow-600' : 'text-green-600';
                    
                    let html = `
                        <div class="text-center mb-6">
                            <h3 class="text-2xl font-bold text-gray-800">Your Stress Analysis</h3>
                        </div>
                        
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                            <div class="bg-white p-6 rounded-lg shadow">
                                <h4 class="text-lg font-semibold text-gray-700 mb-2">Stress Level</h4>
                                <p class="text-3xl font-bold ${{stressColor}}">${{result.stress_level}}/10</p>
                                <p class="text-sm text-gray-600">${{result.category}}</p>
                            </div>
                            
                            <div class="bg-white p-6 rounded-lg shadow">
                                <h4 class="text-lg font-semibold text-gray-700 mb-2">Assessment</h4>
                                <p class="text-gray-800">${{result.description}}</p>
                            </div>
                        </div>
                        
                        <div class="bg-white p-6 rounded-lg shadow">
                            <h4 class="text-lg font-semibold text-gray-700 mb-4">Personalized Recommendations</h4>
                            <ul class="space-y-2">
                    `;
                    
                    result.recommendations.forEach(rec => {{
                        html += `<li class="flex items-start"><span class="text-green-500 mr-2">✓</span><span class="text-gray-700">${{rec}}</span></li>`;
                    }});
                    
                    html += `
                            </ul>
                        </div>
                    `;
                    
                    resultsContent.innerHTML = html;
                    resultsDiv.classList.remove('hidden');
                    resultsDiv.scrollIntoView({{ behavior: 'smooth' }});
                }}
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
        "algorithm": "Rule-based stress assessment",
        "endpoints": ["/", "/health", "/predict", "/docs"]
    }

@app.post("/predict")
async def predict_stress(
    age: int = Form(...),
    sleep_duration: float = Form(...),
    physical_activity_level: int = Form(...),
    heart_rate: int = Form(...),
    quality_of_sleep: int = Form(...)
):
    """Predict stress level using rule-based algorithm"""
    try:
        # Make prediction using rule-based algorithm
        stress_level = simple_stress_prediction(
            age, sleep_duration, physical_activity_level, heart_rate, quality_of_sleep
        )
        
        # Get category and description
        category, description = get_stress_category(stress_level)
        
        # Generate personalized recommendations
        recommendations = get_health_recommendations(
            stress_level, sleep_duration, physical_activity_level, heart_rate, quality_of_sleep
        )
        
        return {
            "stress_level": stress_level,
            "category": category,
            "description": description,
            "recommendations": recommendations,
            "algorithm": "Rule-based analysis",
            "factors_analyzed": [
                "Age and demographic factors",
                "Sleep duration and quality",
                "Physical activity level", 
                "Cardiovascular health (heart rate)",
                "Overall wellness indicators"
            ],
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

@app.get("/api/predict")
async def predict_stress_get(
    age: int,
    sleep_duration: float,
    physical_activity_level: int,
    heart_rate: int,
    quality_of_sleep: int
):
    """GET endpoint for stress prediction"""
    return await predict_stress(
        age, sleep_duration, physical_activity_level, heart_rate, quality_of_sleep
    )

# For Vercel compatibility
def handler(request, response):
    return app
