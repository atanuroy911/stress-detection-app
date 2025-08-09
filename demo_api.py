"""
API Usage Demo Script
Demonstrates how to use the Stress Prediction System API
"""

import requests
import json
from typing import Dict, Any

def predict_stress_level(data: Dict[str, Any], base_url: str = "http://localhost:8000") -> Dict:
    """
    Send prediction request to the API
    
    Args:
        data: Dictionary containing user health information
        base_url: Base URL of the API (default: localhost:8000)
    
    Returns:
        Dictionary containing prediction results
    """
    try:
        response = requests.post(f"{base_url}/predict", data=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return {"error": str(e)}

def get_model_results(base_url: str = "http://localhost:8000") -> Dict:
    """Get model training and testing results"""
    try:
        response = requests.get(f"{base_url}/api/model-results")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting model results: {e}")
        return {"error": str(e)}

def health_check(base_url: str = "http://localhost:8000") -> Dict:
    """Check if the API is running and healthy"""
    try:
        response = requests.get(f"{base_url}/health")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error checking health: {e}")
        return {"error": str(e)}

def demo_predictions():
    """Demonstrate various prediction scenarios"""
    print("🧠 Stress Level Prediction API Demo")
    print("="*50)
    
    # Check if API is running
    print("1. Checking API health...")
    health = health_check()
    if "error" in health:
        print("❌ API is not running. Please start with: python main.py")
        return
    
    print(f"✅ API is healthy! {health['models_loaded']} models loaded")
    
    # Test cases
    test_cases = [
        {
            "name": "👨‍💻 Software Engineer - Moderate Stress",
            "data": {
                'gender': 'Male',
                'age': 30,
                'occupation': 'Software Engineer',
                'sleep_duration': 6.5,
                'quality_of_sleep': 6,
                'physical_activity_level': 45,
                'heart_rate': 75,
                'daily_steps': 5000,
                'blood_pressure': '130/85',
                'bmi_category': 'Normal',
                'sleep_disorder': 'None'
            }
        },
        {
            "name": "👩‍⚕️ Doctor - Low Stress (Healthy Lifestyle)",
            "data": {
                'gender': 'Female',
                'age': 35,
                'occupation': 'Doctor',
                'sleep_duration': 8.0,
                'quality_of_sleep': 9,
                'physical_activity_level': 75,
                'heart_rate': 68,
                'daily_steps': 9000,
                'blood_pressure': '120/78',
                'bmi_category': 'Normal',
                'sleep_disorder': 'None'
            }
        },
        {
            "name": "👔 Manager - High Stress",
            "data": {
                'gender': 'Male',
                'age': 45,
                'occupation': 'Manager',
                'sleep_duration': 5.0,
                'quality_of_sleep': 3,
                'physical_activity_level': 20,
                'heart_rate': 85,
                'daily_steps': 3500,
                'blood_pressure': '145/92',
                'bmi_category': 'Overweight',
                'sleep_disorder': 'Insomnia'
            }
        }
    ]
    
    # Run predictions
    for i, test_case in enumerate(test_cases, 2):
        print(f"\n{i}. Testing: {test_case['name']}")
        print("-" * 40)
        
        result = predict_stress_level(test_case['data'])
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            continue
        
        if result.get('success'):
            # Display results
            ensemble_pred = result.get('ensemble_regression')
            stress_analysis = result.get('stress_analysis', {})
            
            print(f"📊 Ensemble Prediction: {ensemble_pred}/10")
            print(f"🎯 Stress Level: {stress_analysis.get('level', 'Unknown')}")
            print(f"💡 Analysis: {stress_analysis.get('description', 'No description')}")
            
            # Show top 3 regression model predictions
            reg_preds = result.get('regression_predictions', {})
            if reg_preds:
                print("\n🤖 Top Model Predictions:")
                sorted_preds = sorted(reg_preds.items(), key=lambda x: abs(x[1] - ensemble_pred))
                for model, pred in sorted_preds[:3]:
                    print(f"  • {model}: {pred}")
            
            # Show classification results
            clf_preds = result.get('classification_predictions', {})
            if clf_preds:
                print("\n🏷️ Classification Results:")
                for model, pred in clf_preds.items():
                    category = "Low" if pred <= 3 else "Moderate" if pred <= 6 else "High"
                    print(f"  • {model}: {category} ({pred})")
        else:
            print(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
    
    # Get model performance summary
    print(f"\n{len(test_cases)+1}. Model Performance Summary")
    print("-" * 40)
    
    model_results = get_model_results()
    if "error" not in model_results:
        # Show best models
        if 'regression' in model_results:
            best_reg = max(model_results['regression'].items(), 
                          key=lambda x: x[1].get('test_r2', 0))
            print(f"🥇 Best Regression Model: {best_reg[0]} (R² = {best_reg[1].get('test_r2', 0):.4f})")
        
        if 'classification' in model_results:
            best_clf = max(model_results['classification'].items(), 
                          key=lambda x: x[1].get('test_accuracy', 0))
            print(f"🏆 Best Classification Model: {best_clf[0]} (Accuracy = {best_clf[1].get('test_accuracy', 0):.4f})")
        
        if 'unsupervised' in model_results:
            print(f"🔍 Unsupervised Models: {len(model_results['unsupervised'])} algorithms applied")
    
    print("\n" + "="*50)
    print("Demo completed! 🎉")
    print("\nNext steps:")
    print("• Visit http://localhost:8000 for the web interface")
    print("• Check /models page for detailed model comparison")
    print("• Review visualizations in /static directory")

def custom_prediction():
    """Allow user to input custom values for prediction"""
    print("\n🔧 Custom Prediction Mode")
    print("="*30)
    
    # Get user input
    try:
        data = {
            'gender': input("Gender (Male/Female): ").strip(),
            'age': int(input("Age: ")),
            'occupation': input("Occupation: ").strip(),
            'sleep_duration': float(input("Sleep Duration (hours): ")),
            'quality_of_sleep': int(input("Sleep Quality (1-10): ")),
            'physical_activity_level': int(input("Physical Activity Level (1-100): ")),
            'heart_rate': int(input("Heart Rate (bpm): ")),
            'daily_steps': int(input("Daily Steps: ")),
            'blood_pressure': input("Blood Pressure (e.g., 120/80): ").strip(),
            'bmi_category': input("BMI Category (Normal/Overweight/Obese): ").strip(),
            'sleep_disorder': input("Sleep Disorder (None/Insomnia/Sleep Apnea): ").strip()
        }
        
        print("\n🔮 Making prediction...")
        result = predict_stress_level(data)
        
        if result.get('success'):
            ensemble_pred = result.get('ensemble_regression')
            stress_analysis = result.get('stress_analysis', {})
            
            print(f"\n📊 Your Predicted Stress Level: {ensemble_pred}/10")
            print(f"🎯 Category: {stress_analysis.get('level', 'Unknown')}")
            print(f"💡 Recommendation: {stress_analysis.get('description', 'No recommendation')}")
        else:
            print(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
    
    except (ValueError, KeyboardInterrupt) as e:
        print(f"\n❌ Input error: {e}")
        print("Please try again with valid inputs.")

if __name__ == "__main__":
    try:
        # Run demo
        demo_predictions()
        
        # Ask if user wants to try custom prediction
        while True:
            choice = input("\nWould you like to try a custom prediction? (y/n): ").strip().lower()
            if choice == 'y':
                custom_prediction()
            elif choice == 'n':
                print("Thanks for using the Stress Prediction System! 👋")
                break
            else:
                print("Please enter 'y' or 'n'")
    
    except KeyboardInterrupt:
        print("\n\nDemo interrupted. Goodbye! 👋")
