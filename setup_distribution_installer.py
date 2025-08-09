"""
Simple Setup script for Stress Level Prediction App
Creates a distributable package for cross-platform deployment
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import zipfile
import json
from datetime import datetime

def create_standalone_package():
    """Create a standalone package with all dependencies"""
    
    print("Creating Stress Level Prediction App Package...")
    print("="*60)
    
    # Package information
    package_info = {
        "name": "StressLevelPredictionApp",
        "version": "1.0.0",
        "description": "Cross-platform ML application for stress level prediction",
        "author": "ML Course Project",
        "created": datetime.now().isoformat(),
        "platform": sys.platform,
        "python_version": sys.version
    }
    
    # Create package directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_dir = f"StressLevelPredictionApp_v1.0_{timestamp}"
    
    if os.path.exists(package_dir):
        shutil.rmtree(package_dir)
    os.makedirs(package_dir)
    
    print(f"Package directory: {package_dir}")
    
    # Copy main application
    shutil.copy2("stress_prediction_app.py", package_dir)
    print("[OK] Copied main application")
    
    # Copy dataset if available
    if os.path.exists("dataset"):
        shutil.copytree("dataset", os.path.join(package_dir, "dataset"))
        print("[OK] Copied dataset")
    
    # Copy existing models if available
    if os.path.exists("models"):
        shutil.copytree("models", os.path.join(package_dir, "models"))
        print("[OK] Copied pre-trained models")
    
    # Create requirements file for the standalone app
    standalone_requirements = [
        "pandas>=2.0.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "joblib>=1.0.0",
        "scipy>=1.7.0"
    ]
    
    with open(os.path.join(package_dir, "requirements.txt"), 'w') as f:
        f.write("\n".join(standalone_requirements))
    print("[OK] Created requirements.txt")
    
    # Create installation script for Windows
    windows_installer = '''@echo off
echo Installing Stress Level Prediction App...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python found. Installing dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Installation completed successfully!
echo.
echo To run the application, double-click on "run_app.bat"
echo or run: python stress_prediction_app.py
echo.
pause
'''
    
    with open(os.path.join(package_dir, "install.bat"), 'w') as f:
        f.write(windows_installer)
    print("[OK] Created Windows installer")
    
    # Create run script for Windows
    windows_runner = '''@echo off
echo Starting Stress Level Prediction App...
python stress_prediction_app.py
if errorlevel 1 (
    echo.
    echo ERROR: Failed to start the application
    echo Please make sure you have run install.bat first
    pause
)
'''
    
    with open(os.path.join(package_dir, "run_app.bat"), 'w') as f:
        f.write(windows_runner)
    print("[OK] Created Windows runner")
    
    # Create installation script for Unix/Linux/macOS
    unix_installer = '''#!/bin/bash
echo "Installing Stress Level Prediction App..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "ERROR: Python is not installed"
    echo "Please install Python 3.8+ from https://python.org"
    exit 1
fi

# Determine Python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    PYTHON_CMD=python
fi

echo "Python found. Installing dependencies..."
$PYTHON_CMD -m pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Failed to install dependencies"
    echo "Please check your internet connection and try again"
    exit 1
fi

# Make run script executable
chmod +x run_app.sh

echo ""
echo "[SUCCESS] Installation completed successfully!"
echo ""
echo "To run the application:"
echo "  ./run_app.sh"
echo "or:"
echo "  $PYTHON_CMD stress_prediction_app.py"
echo ""
'''
    
    with open(os.path.join(package_dir, "install.sh"), 'w') as f:
        f.write(unix_installer)
    
    # Make installer executable
    os.chmod(os.path.join(package_dir, "install.sh"), 0o755)
    print("[OK] Created Unix/Linux/macOS installer")
    
    # Create run script for Unix/Linux/macOS
    unix_runner = '''#!/bin/bash
echo "Starting Stress Level Prediction App..."

# Determine Python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    PYTHON_CMD=python
fi

$PYTHON_CMD stress_prediction_app.py

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Failed to start the application"
    echo "Please make sure you have run ./install.sh first"
fi
'''
    
    with open(os.path.join(package_dir, "run_app.sh"), 'w') as f:
        f.write(unix_runner)
    
    # Make runner executable
    os.chmod(os.path.join(package_dir, "run_app.sh"), 0o755)
    print("[OK] Created Unix/Linux/macOS runner")
    
    # Create comprehensive README
    readme_content = f"""# Stress Level Prediction App v1.0

A comprehensive machine learning application for predicting stress levels

## Features

- 15+ Machine Learning Models: Regression, Classification, and Unsupervised Learning
- Advanced Feature Engineering: Automatic correlation analysis and feature selection
- User-Friendly GUI: Cross-platform interface built with Tkinter
- Real-time Predictions: Get instant stress level assessments
- Model Visualization: Compare performance with interactive charts
- Export Capabilities: Save models, results, and visualizations
- Health Recommendations: Personalized advice based on predictions

## Quick Start

### Windows Users:
1. Double-click `install.bat` to install dependencies
2. Double-click `run_app.bat` to start the application

### macOS/Linux Users:
1. Open terminal in this directory
2. Run: `./install.sh` to install dependencies
3. Run: `./run_app.sh` to start the application

### Manual Installation:
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python stress_prediction_app.py
```

## System Requirements

- Python: 3.8 or higher
- Operating System: Windows 10+, macOS 10.12+, or Linux
- RAM: Minimum 2GB, Recommended 4GB+
- Disk Space: ~500MB for full installation
- Internet: Required for initial dependency installation

## Usage Instructions

### 1. Data Tab
- Load your own CSV dataset or use the built-in sample data
- View dataset information and preview data
- Export results when training is complete

### 2. Training Tab
- Select which models to train (15+ available)
- Run feature engineering to optimize predictions
- Monitor training progress in real-time
- View detailed training logs and model analysis

### 3. Prediction Tab
- Fill out the health assessment form
- Get instant stress level predictions from all models
- View ensemble results and individual model outputs
- Receive personalized health recommendations

### 4. Results Tab
- Visualize model performance comparisons
- View correlation matrices and feature importance
- Save charts and analysis results
- Export complete model packages

## Included Models

### Regression Models (9):
- Linear Regression, Ridge Regression, Lasso Regression
- Random Forest Regressor, Gradient Boosting Regressor
- Support Vector Regression (SVR), Multi-layer Perceptron (MLP)
- Decision Tree Regressor, K-Nearest Neighbors (KNN)

### Classification Models (3):
- Logistic Regression, Random Forest Classifier
- Support Vector Classifier (SVC)

### Unsupervised Models (3):
- K-Means Clustering, Hierarchical Clustering
- Principal Component Analysis (PCA)

## Input Parameters

The application analyzes the following health factors:
- Demographics: Gender, Age, Occupation
- Sleep Metrics: Duration, Quality, Sleep Disorders
- Physical Health: Activity Level, Heart Rate, Daily Steps, BMI
- Vital Signs: Blood Pressure measurements

## Performance Metrics

- Regression: R² Score, MSE, RMSE, MAE, Cross-validation
- Classification: Accuracy, Precision, Recall, F1-Score
- Unsupervised: Cluster Analysis, Explained Variance

## Data Format

For custom datasets, use CSV format with these columns:
```
Person ID, Gender, Age, Occupation, Sleep Duration, Quality of Sleep,
Physical Activity Level, Stress Level, BMI Category, Blood Pressure,
Heart Rate, Daily Steps, Sleep Disorder
```

## Troubleshooting

### Common Issues:

**Installation fails:**
- Ensure Python 3.8+ is installed and in PATH
- Try: `python -m pip install --upgrade pip`
- Check internet connection

**Application won't start:**
- Verify all dependencies are installed: `pip list`
- Try running: `python stress_prediction_app.py` from command line

**Training fails:**
- Check that dataset has required columns
- Ensure sufficient system memory (2GB+)
- Verify no missing critical data

**Predictions return errors:**
- Fill all required form fields
- Ensure models are trained first
- Check input value ranges

### Getting Help:
1. Check the training log in the application for detailed error messages
2. Verify system requirements are met
3. Try with sample data first to test functionality

## Important Disclaimers

- This application is for educational and informational purposes only
- Predictions should not replace professional medical advice
- Always consult healthcare professionals for medical concerns
- Results may vary based on input data quality and completeness

## Technical Details

### Architecture:
- GUI Framework: Tkinter (built into Python)
- ML Library: Scikit-learn
- Data Processing: Pandas, NumPy
- Visualization: Matplotlib, Seaborn
- Model Persistence: Joblib

### File Structure:
```
StressLevelPredictionApp/
├── stress_prediction_app.py    # Main application
├── requirements.txt            # Python dependencies
├── install.bat/.sh            # Installation scripts
├── run_app.bat/.sh            # Application launchers
├── dataset/                   # Sample data (if included)
├── models/                    # Pre-trained models (if included)
└── README.md                  # This file
```

## Updates and Support

Version: 1.0.0
Created: {package_info['created']}
Platform: {package_info['platform']}
Python: {package_info['python_version']}

For updates and support, check the training logs within the application for detailed information.

## License

This project is open source and available under the MIT License.

---

Thank you for using the Stress Level Prediction App!

Built with care for machine learning education and healthcare applications
"""
    
    with open(os.path.join(package_dir, "README.md"), 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print("[OK] Created comprehensive README")
    
    # Save package info
    with open(os.path.join(package_dir, "package_info.json"), 'w') as f:
        json.dump(package_info, f, indent=2)
    print("[OK] Saved package information")
    
    # Create ZIP distribution
    zip_filename = f"{package_dir}.zip"
    print(f"Creating ZIP distribution: {zip_filename}")
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(package_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arc_path = os.path.relpath(file_path, package_dir)
                zipf.write(file_path, arc_path)
    
    print("[OK] ZIP distribution created")
    
    # Summary
    print("\n" + "="*60)
    print("PACKAGE CREATION COMPLETED!")
    print("="*60)
    print(f"Package Directory: {package_dir}")
    print(f"ZIP Distribution: {zip_filename}")
    print(f"Package Size: {get_dir_size(package_dir):.1f} MB")
    print(f"Total Files: {count_files(package_dir)}")
    
    print("\nDistribution Contents:")
    print("• stress_prediction_app.py - Main application")
    print("• requirements.txt - Python dependencies")  
    print("• install.bat/.sh - Installation scripts")
    print("• run_app.bat/.sh - Application launchers")
    print("• README.md - Comprehensive documentation")
    print("• package_info.json - Package metadata")
    if os.path.exists(os.path.join(package_dir, "dataset")):
        print("• dataset/ - Sample training data")
    if os.path.exists(os.path.join(package_dir, "models")):
        print("• models/ - Pre-trained ML models")
    
    print(f"\nREADY FOR DISTRIBUTION!")
    print(f"Users can simply:")
    print(f"1. Extract {zip_filename}")
    print(f"2. Run install script for their platform")
    print(f"3. Launch the application")
    
    return package_dir, zip_filename

def get_dir_size(directory):
    """Calculate directory size in MB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)

def count_files(directory):
    """Count total files in directory"""
    count = 0
    for root, dirs, files in os.walk(directory):
        count += len(files)
    return count

def main():
    """Main setup function"""
    print("Stress Level Prediction App - Setup & Distribution Tool")
    print("="*70)
    
    # Check if we're in the right directory
    if not os.path.exists("stress_prediction_app.py"):
        print("ERROR: stress_prediction_app.py not found!")
        print("Please run this script from the same directory as the main application.")
        return
    
    try:
        # Create standalone package
        package_dir, zip_file = create_standalone_package()
        
        print(f"\nDISTRIBUTION READY!")
        print(f"Share: {zip_file}")
        print(f"Test: {package_dir}")
        
    except Exception as e:
        print(f"Setup failed: {e}")
        return

if __name__ == "__main__":
    main()
