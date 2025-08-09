# Stress Level Prediction System

A comprehensive machine learning system for predicting stress levels using various supervised and unsupervised learning algorithms with advanced feature engineering and a modern web interface.

## 🚀 Features

### Machine Learning Models
- **9 Regression Models**: Linear, Ridge, Lasso, Random Forest, Gradient Boosting, SVR, MLP, Decision Tree, KNN
- **3 Classification Models**: Logistic Regression, Random Forest Classifier, SVC
- **3 Unsupervised Models**: K-Means Clustering, Hierarchical Clustering, PCA

### Advanced Feature Engineering
- **Correlation Matrix Analysis**: Automated feature selection based on correlation with target variable
- **Engineered Features**: 
  - Sleep Quality Score (Quality × Duration)
  - Activity-Sleep Ratio
  - Steps per Hour Awake
  - Blood Pressure Risk Assessment
  - Heart Rate Categories
  - BMI Risk Scoring
  - Age Grouping

### Modern Web Interface
- **FastAPI Backend**: High-performance API with automatic documentation
- **Responsive Design**: Modern UI using Tailwind CSS
- **Interactive Predictions**: Real-time stress level prediction
- **Model Comparison Dashboard**: Comprehensive visualization of model performance
- **Health Recommendations**: Personalized suggestions based on prediction results

## 📊 Model Performance

### Best Performing Models
- **Regression**: Decision Tree Regressor (R² = 1.0000)
- **Classification**: Random Forest Classifier (Accuracy = 100%)

### Feature Importance (Top Contributors)
1. Quality of Sleep (r = -0.898)
2. Sleep Quality Score (r = -0.886)
3. Sleep Duration (r = -0.811)
4. Heart Rate (r = 0.670)
5. Age (r = 0.422)

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
1. **Clone the repository**
   ```bash
   cd "ML COURSE/app2-stress"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models**
   ```bash
   python train.py
   ```

4. **Test the models**
   ```bash
   python test.py
   ```

5. **Run the web application**
   ```bash
   python main.py
   ```

6. **Access the application**
   Open http://localhost:8000 in your browser

## 📁 Project Structure

```
app2-stress/
├── train.py                 # Model training script
├── test.py                  # Model testing script
├── main.py                  # FastAPI web application
├── requirements.txt         # Python dependencies
├── models/                  # Trained models and preprocessors
│   ├── *.pkl               # Saved ML models
│   ├── results.json        # Training results
│   └── test_results.json   # Testing results
├── static/                  # Generated visualizations
│   ├── correlation_matrix.png
│   ├── model_comparison.png
│   ├── pca_analysis.png
│   └── *.png              # Various charts and plots
├── templates/              # HTML templates
│   ├── index.html         # Main prediction interface
│   └── models.html        # Model comparison dashboard
└── dataset/               # Training data
    └── Sleep_health_and_lifestyle_dataset.csv
```

## 🔬 Technical Implementation

### Feature Engineering Pipeline
1. **Data Loading & Exploration**: Comprehensive dataset analysis
2. **Feature Creation**: 
   - Blood pressure parsing (systolic/diastolic)
   - Composite health scores
   - Risk categorization
   - Categorical encoding
3. **Correlation Analysis**: Automated feature selection (threshold > 0.1)
4. **Scaling**: StandardScaler for numerical features

### Model Training Process
1. **Data Splitting**: 80/20 train-test split with stratification
2. **Cross-Validation**: 5-fold CV for model validation
3. **Hyperparameter Tuning**: Grid search for optimal parameters
4. **Model Persistence**: Joblib serialization for deployment

### Web Application Architecture
- **Backend**: FastAPI with async support
- **Frontend**: Tailwind CSS + Vanilla JavaScript
- **API Endpoints**:
  - `POST /predict`: Real-time prediction
  - `GET /models`: Model comparison page
  - `GET /api/model-results`: JSON results API
  - `GET /health`: System health check

## 📈 Model Results

### Regression Performance
| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Decision Tree | 1.0000 | 0.0000 | 0.0000 |
| Gradient Boosting | 0.9989 | 0.0595 | 0.0264 |
| Random Forest | 0.9955 | 0.1189 | 0.0322 |
| MLP | 0.9882 | 0.1921 | 0.0781 |
| SVR | 0.9782 | 0.2619 | 0.1139 |

### Classification Performance
| Model | Test Accuracy | CV Accuracy |
|-------|---------------|-------------|
| Random Forest | 100.0% | 95.98% |
| Logistic Regression | 94.65% | 91.98% |
| SVC | 94.65% | 93.64% |

### Clustering Analysis
- **K-Means**: 3 clusters representing low (3.0), medium (5.5), high (6.5) stress levels
- **Hierarchical**: Similar clustering pattern with clear stress level separation
- **PCA**: 6 components explain 95% of variance

## 🎯 Usage Examples

### Web Interface
1. Navigate to http://localhost:8000
2. Fill out the health assessment form
3. Click "Predict Stress Level"
4. View comprehensive results from all models
5. Review personalized health recommendations

### API Usage
```python
import requests

# Prediction endpoint
response = requests.post('http://localhost:8000/predict', data={
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
})

results = response.json()
print(f"Predicted Stress Level: {results['ensemble_regression']}")
```

## 🎨 Visualizations

The system automatically generates:
- **Correlation Matrix**: Feature relationships heatmap
- **Model Comparison**: Performance metrics visualization
- **PCA Analysis**: Dimensionality reduction plots
- **Clustering Results**: K-means elbow curve
- **Confusion Matrices**: Classification model evaluation
- **Feature Importance**: Top contributing factors

## 🏥 Health Recommendations

Based on predicted stress levels:

### Low Stress (≤3)
- ✅ Maintain current healthy lifestyle
- ✅ Continue regular exercise
- ✅ Keep monitoring stress levels

### Moderate Stress (4-6)
- ⚠️ Implement stress management techniques
- ⚠️ Try meditation or mindfulness
- ⚠️ Improve sleep quality and duration
- ⚠️ Increase physical activity

### High Stress (≥7)
- 🚨 Consider consulting healthcare professional
- 🚨 Implement immediate stress reduction strategies
- 🚨 Prioritize sleep and relaxation
- 🚨 Consider professional counseling

## ⚠️ Important Disclaimer

This tool is for **informational purposes only** and should not replace professional medical advice. If you're experiencing high stress levels or health concerns, please consult with a qualified healthcare professional.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙋‍♂️ Support

For questions, issues, or suggestions:
1. Check the model comparison dashboard for performance insights
2. Review the generated visualizations in the `/static` directory
3. Examine the detailed logs in the console output

## 🔄 Recent Updates

- ✨ Added ensemble prediction combining multiple models
- 🎨 Improved UI with modern Tailwind CSS design
- 📊 Enhanced visualizations with Chart.js integration
- 🚀 Optimized model performance with advanced feature engineering
- 🔒 Added comprehensive error handling and validation

---

**Built with ❤️ using Python, FastAPI, Scikit-learn, and modern web technologies**
# stress-detection-app
