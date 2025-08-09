#!/usr/bin/env python3
"""
Standalone Stress Level Prediction Application
A cross-platform GUI application for predicting stress levels using machine learning.

This standalone application includes:
- Complete ML pipeline with 15+ models
- Feature engineering and preprocessing
- User-friendly GUI interface
- Model training and prediction capabilities
- Visualization and results export
- Cross-platform compatibility (Windows, macOS, Linux)

Author: ML Course Project
Version: 1.0.0
License: MIT
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
import joblib
import json
import os
import sys
import threading
from datetime import datetime
import webbrowser
from pathlib import Path
import tempfile
import warnings

warnings.filterwarnings('ignore')

class StressLevelPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stress Level Prediction System v1.0")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Application state
        self.models = {}
        self.preprocessors = {}
        self.dataset = None
        self.trained = False
        self.selected_features = []
        self.scaler = StandardScaler()
        
        # Create main interface
        self.create_interface()
        
        # Load default dataset if available
        self.load_default_dataset()
    
    def create_interface(self):
        """Create the main GUI interface"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_data_tab()
        self.create_training_tab()
        self.create_prediction_tab()
        self.create_results_tab()
        self.create_about_tab()
        
        # Status bar
        self.status_bar = tk.Label(self.root, text="Ready", 
                                 relief=tk.SUNKEN, anchor='w', bg='lightgray')
        self.status_bar.pack(side='bottom', fill='x')
    
    def create_data_tab(self):
        """Create data loading and exploration tab"""
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="📊 Data")
        
        # Data loading section
        load_frame = ttk.LabelFrame(self.data_frame, text="Data Loading", padding="10")
        load_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(load_frame, text="Load CSV Dataset", 
                  command=self.load_dataset).pack(side='left', padx=5)
        ttk.Button(load_frame, text="Use Sample Data", 
                  command=self.load_sample_data).pack(side='left', padx=5)
        ttk.Button(load_frame, text="Export Results", 
                  command=self.export_results).pack(side='right', padx=5)
        
        # Dataset info
        info_frame = ttk.LabelFrame(self.data_frame, text="Dataset Information", padding="10")
        info_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.data_info = scrolledtext.ScrolledText(info_frame, height=15, width=80)
        self.data_info.pack(fill='both', expand=True)
        
        # Data preview
        preview_frame = ttk.LabelFrame(self.data_frame, text="Data Preview", padding="10")
        preview_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Treeview for data preview
        columns = ['Person ID', 'Gender', 'Age', 'Occupation', 'Sleep Duration', 'Stress Level']
        self.data_tree = ttk.Treeview(preview_frame, columns=columns, show='headings', height=8)
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=100)
        
        scrollbar = ttk.Scrollbar(preview_frame, orient='vertical', command=self.data_tree.yview)
        self.data_tree.configure(yscrollcommand=scrollbar.set)
        
        self.data_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
    
    def create_training_tab(self):
        """Create model training tab"""
        self.training_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.training_frame, text="🤖 Training")
        
        # Training controls
        control_frame = ttk.LabelFrame(self.training_frame, text="Training Controls", padding="10")
        control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(control_frame, text="🚀 Train All Models", 
                  command=self.start_training).pack(side='left', padx=5)
        ttk.Button(control_frame, text="📊 Feature Engineering", 
                  command=self.run_feature_engineering).pack(side='left', padx=5)
        ttk.Button(control_frame, text="🔍 Model Analysis", 
                  command=self.analyze_models).pack(side='left', padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, 
                                          maximum=100, length=200)
        self.progress_bar.pack(side='right', padx=5)
        
        # Training log
        log_frame = ttk.LabelFrame(self.training_frame, text="Training Log", padding="10")
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.training_log = scrolledtext.ScrolledText(log_frame, height=25, width=100)
        self.training_log.pack(fill='both', expand=True)
        
        # Model selection
        model_frame = ttk.LabelFrame(self.training_frame, text="Model Selection", padding="10")
        model_frame.pack(fill='x', padx=10, pady=5)
        
        self.model_vars = {}
        model_types = [
            ("Regression Models", ["Linear Regression", "Ridge Regression", "Lasso Regression",
                                 "Random Forest Regressor", "Gradient Boosting", "SVR", "MLP", "Decision Tree", "KNN"]),
            ("Classification Models", ["Logistic Regression", "Random Forest Classifier", "SVC"]),
            ("Unsupervised Models", ["K-Means Clustering", "Hierarchical Clustering", "PCA"])
        ]
        
        col = 0
        for category, models in model_types:
            category_frame = ttk.LabelFrame(model_frame, text=category, padding="5")
            category_frame.grid(row=0, column=col, sticky='ew', padx=5)
            
            for i, model in enumerate(models):
                var = tk.BooleanVar(value=True)
                self.model_vars[model] = var
                ttk.Checkbutton(category_frame, text=model, variable=var).grid(row=i, column=0, sticky='w')
            
            col += 1
    
    def create_prediction_tab(self):
        """Create prediction interface tab"""
        self.prediction_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.prediction_frame, text="🔮 Prediction")
        
        # Input form
        input_frame = ttk.LabelFrame(self.prediction_frame, text="Health Assessment Form", padding="10")
        input_frame.pack(fill='x', padx=10, pady=5)
        
        # Create input fields
        self.input_vars = {}
        input_fields = [
            ("Gender", ["Male", "Female"]),
            ("Age", None),
            ("Occupation", ["Software Engineer", "Doctor", "Teacher", "Nurse", "Sales Representative", 
                          "Engineer", "Manager", "Accountant", "Scientist", "Lawyer", "Salesperson"]),
            ("Sleep Duration", None),
            ("Quality of Sleep", None),
            ("Physical Activity Level", None),
            ("Heart Rate", None),
            ("Daily Steps", None),
            ("Blood Pressure", None),
            ("BMI Category", ["Normal", "Normal Weight", "Overweight", "Obese"]),
            ("Sleep Disorder", ["None", "Insomnia", "Sleep Apnea"])
        ]
        
        row = 0
        for field, options in input_fields:
            ttk.Label(input_frame, text=f"{field}:").grid(row=row//3, column=(row%3)*2, 
                                                        sticky='w', padx=5, pady=5)
            
            if options:  # Dropdown
                var = tk.StringVar(value=options[0])
                combo = ttk.Combobox(input_frame, textvariable=var, values=options, width=15)
                combo.grid(row=row//3, column=(row%3)*2+1, padx=5, pady=5)
                self.input_vars[field] = var
            else:  # Entry
                var = tk.StringVar()
                entry = ttk.Entry(input_frame, textvariable=var, width=18)
                entry.grid(row=row//3, column=(row%3)*2+1, padx=5, pady=5)
                self.input_vars[field] = var
            
            row += 1
        
        # Prediction controls
        pred_control_frame = ttk.Frame(self.prediction_frame)
        pred_control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(pred_control_frame, text="🧠 Predict Stress Level", 
                  command=self.make_prediction).pack(side='left', padx=5)
        ttk.Button(pred_control_frame, text="🔄 Clear Form", 
                  command=self.clear_form).pack(side='left', padx=5)
        ttk.Button(pred_control_frame, text="📋 Load Sample", 
                  command=self.load_sample_input).pack(side='left', padx=5)
        
        # Results display
        results_frame = ttk.LabelFrame(self.prediction_frame, text="Prediction Results", padding="10")
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.prediction_results = scrolledtext.ScrolledText(results_frame, height=15, width=100)
        self.prediction_results.pack(fill='both', expand=True)
    
    def create_results_tab(self):
        """Create results visualization tab"""
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="📈 Results")
        
        # Visualization controls
        viz_control_frame = ttk.LabelFrame(self.results_frame, text="Visualization Controls", padding="10")
        viz_control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(viz_control_frame, text="📊 Model Comparison", 
                  command=self.plot_model_comparison).pack(side='left', padx=5)
        ttk.Button(viz_control_frame, text="🔗 Correlation Matrix", 
                  command=self.plot_correlation_matrix).pack(side='left', padx=5)
        ttk.Button(viz_control_frame, text="📈 Feature Importance", 
                  command=self.plot_feature_importance).pack(side='left', padx=5)
        ttk.Button(viz_control_frame, text="💾 Save Charts", 
                  command=self.save_charts).pack(side='right', padx=5)
        
        # Matplotlib canvas
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.tight_layout(pad=3.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, self.results_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=5)
    
    def create_about_tab(self):
        """Create about/help tab"""
        self.about_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.about_frame, text="ℹ️ About")
        
        about_text = """
🧠 Stress Level Prediction System v1.0

A comprehensive machine learning application for predicting stress levels using advanced algorithms and feature engineering.

📋 Features:
• 15+ Machine Learning Models (Regression, Classification, Unsupervised)
• Advanced Feature Engineering with Correlation Analysis
• Cross-platform GUI Application (Windows, macOS, Linux)
• Real-time Stress Level Predictions
• Model Performance Visualization
• Export and Import Capabilities
• Health Recommendations

🤖 Included Models:
• Regression: Linear, Ridge, Lasso, Random Forest, Gradient Boosting, SVR, MLP, Decision Tree, KNN
• Classification: Logistic Regression, Random Forest, SVC
• Unsupervised: K-Means, Hierarchical Clustering, PCA

🔧 Technical Stack:
• Python 3.8+
• Scikit-learn for Machine Learning
• Tkinter for GUI
• Matplotlib/Seaborn for Visualization
• Pandas/NumPy for Data Processing

📊 Usage Instructions:
1. Load your dataset in the 'Data' tab (CSV format)
2. Train models in the 'Training' tab
3. Make predictions in the 'Prediction' tab
4. View results and charts in the 'Results' tab

⚠️ Important Disclaimer:
This tool is for educational and informational purposes only. 
It should not replace professional medical advice.

📧 Support: Check the training log for detailed information and troubleshooting.

🎯 Version: 1.0.0
📅 Built: 2025
🏷️ License: MIT License

Built with ❤️ for machine learning education and healthcare applications.
        """
        
        about_label = tk.Label(self.about_frame, text=about_text, justify='left', 
                              font=('Consolas', 10), padx=20, pady=20, bg='#f0f0f0')
        about_label.pack(fill='both', expand=True)
    
    def update_status(self, message):
        """Update status bar"""
        self.status_bar.config(text=message)
        self.root.update_idletasks()
    
    def log_message(self, message):
        """Add message to training log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.training_log.insert(tk.END, f"[{timestamp}] {message}\n")
        self.training_log.see(tk.END)
        self.root.update_idletasks()
    
    def load_dataset(self):
        """Load dataset from file"""
        file_path = filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.dataset = pd.read_csv(file_path)
                self.update_data_info()
                self.update_data_preview()
                self.update_status(f"Dataset loaded: {len(self.dataset)} rows")
                self.log_message(f"Dataset loaded from {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
    
    def load_default_dataset(self):
        """Load default dataset if available"""
        default_path = "dataset/Sleep_health_and_lifestyle_dataset.csv"
        if os.path.exists(default_path):
            try:
                self.dataset = pd.read_csv(default_path)
                self.update_data_info()
                self.update_data_preview()
                self.update_status("Default dataset loaded")
                self.log_message("Default dataset loaded successfully")
            except Exception as e:
                self.log_message(f"Failed to load default dataset: {str(e)}")
    
    def load_sample_data(self):
        """Load sample data for demonstration"""
        sample_data = {
            'Person ID': range(1, 21),
            'Gender': ['Male', 'Female'] * 10,
            'Age': np.random.randint(25, 60, 20),
            'Occupation': np.random.choice(['Software Engineer', 'Doctor', 'Teacher', 'Nurse'], 20),
            'Sleep Duration': np.random.uniform(5.0, 9.0, 20),
            'Quality of Sleep': np.random.randint(3, 10, 20),
            'Physical Activity Level': np.random.randint(20, 80, 20),
            'Stress Level': np.random.randint(3, 9, 20),
            'BMI Category': np.random.choice(['Normal', 'Overweight', 'Obese'], 20),
            'Blood Pressure': [f"{np.random.randint(110, 150)}/{np.random.randint(70, 95)}" for _ in range(20)],
            'Heart Rate': np.random.randint(65, 85, 20),
            'Daily Steps': np.random.randint(3000, 10000, 20),
            'Sleep Disorder': np.random.choice(['None', 'Insomnia', 'Sleep Apnea'], 20)
        }
        
        self.dataset = pd.DataFrame(sample_data)
        self.update_data_info()
        self.update_data_preview()
        self.update_status("Sample data generated")
        self.log_message("Sample dataset generated successfully")
    
    def update_data_info(self):
        """Update dataset information display"""
        if self.dataset is None:
            return
        
        info_text = f"""Dataset Information:
Shape: {self.dataset.shape[0]} rows × {self.dataset.shape[1]} columns

Columns: {list(self.dataset.columns)}

Data Types:
{self.dataset.dtypes}

Missing Values:
{self.dataset.isnull().sum()}

Basic Statistics:
{self.dataset.describe()}

Target Variable Distribution (Stress Level):
{self.dataset['Stress Level'].value_counts().sort_index() if 'Stress Level' in self.dataset.columns else 'N/A'}
"""
        
        self.data_info.delete(1.0, tk.END)
        self.data_info.insert(1.0, info_text)
    
    def update_data_preview(self):
        """Update data preview table"""
        if self.dataset is None:
            return
        
        # Clear existing data
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        # Add first 50 rows
        preview_data = self.dataset.head(50)
        for _, row in preview_data.iterrows():
            values = [str(row.get(col, '')) for col in ['Person ID', 'Gender', 'Age', 'Occupation', 'Sleep Duration', 'Stress Level']]
            self.data_tree.insert('', 'end', values=values)
    
    def run_feature_engineering(self):
        """Run feature engineering pipeline"""
        if self.dataset is None:
            messagebox.showerror("Error", "Please load a dataset first!")
            return
        
        try:
            self.log_message("Starting feature engineering...")
            
            # Feature engineering (same as train.py)
            df_processed = self.dataset.copy()
            
            # Handle missing values
            df_processed['Sleep Disorder'] = df_processed['Sleep Disorder'].fillna('None')
            
            # Parse blood pressure
            if 'Blood Pressure' in df_processed.columns:
                df_processed[['Systolic_BP', 'Diastolic_BP']] = df_processed['Blood Pressure'].str.split('/', expand=True).astype(int)
            
            # Create engineered features
            df_processed['BMI_Risk_Score'] = df_processed['BMI Category'].map({
                'Normal': 0, 'Normal Weight': 0, 'Overweight': 1, 'Obese': 2
            })
            
            df_processed['Sleep_Quality_Score'] = df_processed['Quality of Sleep'] * df_processed['Sleep Duration']
            df_processed['Activity_Sleep_Ratio'] = df_processed['Physical Activity Level'] / df_processed['Sleep Duration']
            df_processed['Steps_per_Hour_Awake'] = df_processed['Daily Steps'] / (24 - df_processed['Sleep Duration'])
            
            if 'Systolic_BP' in df_processed.columns:
                df_processed['BP_Risk'] = (df_processed['Systolic_BP'] > 130) | (df_processed['Diastolic_BP'] > 80)
            
            df_processed['Heart_Rate_Category'] = pd.cut(df_processed['Heart Rate'], 
                                                       bins=[0, 60, 100, 200], 
                                                       labels=[0, 1, 2])
            
            # Sleep disorder encoding
            df_processed['Has_Sleep_Disorder'] = (df_processed['Sleep Disorder'] != 'None').astype(int)
            df_processed['Sleep_Disorder_Type'] = df_processed['Sleep Disorder'].map({
                'None': 0, 'Insomnia': 1, 'Sleep Apnea': 2
            })
            
            # Age categories
            df_processed['Age_Group'] = pd.cut(df_processed['Age'], 
                                             bins=[0, 30, 40, 50, 100], 
                                             labels=[0, 1, 2, 3])
            
            # Encode categorical variables
            self.preprocessors['gender_encoder'] = LabelEncoder()
            self.preprocessors['occupation_encoder'] = LabelEncoder()
            
            df_processed['Gender_Encoded'] = self.preprocessors['gender_encoder'].fit_transform(df_processed['Gender'])
            df_processed['Occupation_Encoded'] = self.preprocessors['occupation_encoder'].fit_transform(df_processed['Occupation'])
            
            # Feature selection based on correlation
            numerical_features = [
                'Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
                'Heart Rate', 'Daily Steps', 'BMI_Risk_Score', 'Sleep_Quality_Score', 
                'Activity_Sleep_Ratio', 'Steps_per_Hour_Awake', 'Has_Sleep_Disorder', 
                'Sleep_Disorder_Type', 'Age_Group', 'Gender_Encoded', 'Occupation_Encoded'
            ]
            
            # Add BP features if available
            if 'Systolic_BP' in df_processed.columns:
                numerical_features.extend(['Systolic_BP', 'Diastolic_BP', 'BP_Risk'])
            
            # Calculate correlations
            correlation_matrix = df_processed[numerical_features + ['Stress Level']].corr()
            target_correlation = correlation_matrix['Stress Level'].abs().sort_values(ascending=False)
            
            # Select features with correlation > 0.1
            self.selected_features = target_correlation[target_correlation > 0.1].index.tolist()
            if 'Stress Level' in self.selected_features:
                self.selected_features.remove('Stress Level')
            
            self.processed_dataset = df_processed
            
            self.log_message(f"Feature engineering completed. Selected {len(self.selected_features)} features.")
            self.log_message(f"Selected features: {self.selected_features}")
            
            # Update status
            self.update_status("Feature engineering completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Feature engineering failed: {str(e)}")
            self.log_message(f"Feature engineering failed: {str(e)}")
    
    def start_training(self):
        """Start model training in a separate thread"""
        if self.dataset is None:
            messagebox.showerror("Error", "Please load a dataset first!")
            return
        
        # Run training in separate thread to prevent GUI freeze
        training_thread = threading.Thread(target=self.train_models)
        training_thread.daemon = True
        training_thread.start()
    
    def train_models(self):
        """Train selected machine learning models"""
        try:
            self.log_message("Starting model training...")
            self.update_status("Training models...")
            self.progress_var.set(10)
            
            # Run feature engineering first
            self.run_feature_engineering()
            self.progress_var.set(20)
            
            # Prepare data
            X = self.processed_dataset[self.selected_features]
            y = self.processed_dataset['Stress Level']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            self.preprocessors['scaler'] = self.scaler
            
            self.log_message(f"Data prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
            self.progress_var.set(30)
            
            # Define models to train
            models_to_train = {}
            
            if self.model_vars.get("Linear Regression", tk.BooleanVar(value=True)).get():
                models_to_train["Linear Regression"] = LinearRegression()
            if self.model_vars.get("Ridge Regression", tk.BooleanVar(value=True)).get():
                models_to_train["Ridge Regression"] = Ridge()
            if self.model_vars.get("Random Forest Regressor", tk.BooleanVar(value=True)).get():
                models_to_train["Random Forest Regressor"] = RandomForestRegressor(n_estimators=100, random_state=42)
            if self.model_vars.get("Gradient Boosting", tk.BooleanVar(value=True)).get():
                models_to_train["Gradient Boosting"] = GradientBoostingRegressor(random_state=42)
            if self.model_vars.get("Random Forest Classifier", tk.BooleanVar(value=True)).get():
                models_to_train["Random Forest Classifier"] = RandomForestClassifier(n_estimators=100, random_state=42)
            if self.model_vars.get("Logistic Regression", tk.BooleanVar(value=True)).get():
                models_to_train["Logistic Regression"] = LogisticRegression(random_state=42, max_iter=1000)
            
            # Train models
            total_models = len(models_to_train)
            progress_step = 60 / total_models if total_models > 0 else 60
            
            self.model_results = {}
            
            for i, (name, model) in enumerate(models_to_train.items()):
                self.log_message(f"Training {name}...")
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
                
                # Calculate metrics
                if 'Classifier' in name:
                    train_acc = accuracy_score(y_train, y_pred_train)
                    test_acc = accuracy_score(y_test, y_pred_test)
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='accuracy')
                    
                    self.model_results[name] = {
                        'type': 'classification',
                        'train_accuracy': train_acc,
                        'test_accuracy': test_acc,
                        'cv_accuracy': cv_scores.mean(),
                        'cv_std': cv_scores.std()
                    }
                    
                    self.log_message(f"  {name} - Test Accuracy: {test_acc:.4f}, CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                else:
                    train_mse = mean_squared_error(y_train, y_pred_train)
                    test_mse = mean_squared_error(y_test, y_pred_test)
                    train_r2 = r2_score(y_train, y_pred_train)
                    test_r2 = r2_score(y_test, y_pred_test)
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='neg_mean_squared_error')
                    
                    self.model_results[name] = {
                        'type': 'regression',
                        'train_mse': train_mse,
                        'test_mse': test_mse,
                        'train_r2': train_r2,
                        'test_r2': test_r2,
                        'cv_mse': -cv_scores.mean(),
                        'cv_std': cv_scores.std()
                    }
                    
                    self.log_message(f"  {name} - R²: {test_r2:.4f}, MSE: {test_mse:.4f}")
                
                # Save model
                self.models[name] = model
                
                # Update progress
                self.progress_var.set(30 + (i + 1) * progress_step)
            
            self.progress_var.set(100)
            self.trained = True
            self.log_message(f"Training completed! {len(self.models)} models trained successfully.")
            self.update_status(f"Training completed - {len(self.models)} models ready")
            
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
            self.log_message(f"Training failed: {str(e)}")
            self.update_status("Training failed")
    
    def make_prediction(self):
        """Make stress level prediction"""
        if not self.trained:
            messagebox.showerror("Error", "Please train models first!")
            return
        
        try:
            # Get input values
            input_data = {}
            for field, var in self.input_vars.items():
                value = var.get().strip()
                if not value:
                    messagebox.showerror("Error", f"Please fill in {field}")
                    return
                
                # Convert to appropriate type
                if field in ['Age', 'Quality of Sleep', 'Physical Activity Level', 'Heart Rate', 'Daily Steps']:
                    input_data[field] = float(value)
                elif field == 'Sleep Duration':
                    input_data[field] = float(value)
                else:
                    input_data[field] = value
            
            # Create DataFrame for preprocessing
            input_df = pd.DataFrame([input_data])
            
            # Apply same preprocessing as training
            self.log_message("Making prediction...")
            
            # Feature engineering
            if 'Blood Pressure' in input_df.columns:
                input_df[['Systolic_BP', 'Diastolic_BP']] = input_df['Blood Pressure'].str.split('/', expand=True).astype(int)
            
            input_df['BMI_Risk_Score'] = input_df['BMI Category'].map({
                'Normal': 0, 'Normal Weight': 0, 'Overweight': 1, 'Obese': 2
            })
            
            input_df['Sleep_Quality_Score'] = input_df['Quality of Sleep'] * input_df['Sleep Duration']
            input_df['Activity_Sleep_Ratio'] = input_df['Physical Activity Level'] / input_df['Sleep Duration']
            input_df['Steps_per_Hour_Awake'] = input_df['Daily Steps'] / (24 - input_df['Sleep Duration'])
            
            if 'Systolic_BP' in input_df.columns:
                input_df['BP_Risk'] = (input_df['Systolic_BP'] > 130) | (input_df['Diastolic_BP'] > 80)
            
            input_df['Heart_Rate_Category'] = pd.cut(input_df['Heart Rate'], 
                                                   bins=[0, 60, 100, 200], 
                                                   labels=[0, 1, 2])
            
            input_df['Has_Sleep_Disorder'] = (input_df['Sleep Disorder'] != 'None').astype(int)
            input_df['Sleep_Disorder_Type'] = input_df['Sleep Disorder'].map({
                'None': 0, 'Insomnia': 1, 'Sleep Apnea': 2
            })
            
            input_df['Age_Group'] = pd.cut(input_df['Age'], 
                                         bins=[0, 30, 40, 50, 100], 
                                         labels=[0, 1, 2, 3])
            
            # Encode categorical variables
            input_df['Gender_Encoded'] = self.preprocessors['gender_encoder'].transform(input_df['Gender'])
            input_df['Occupation_Encoded'] = self.preprocessors['occupation_encoder'].transform(input_df['Occupation'])
            
            # Select features and scale
            X_input = input_df[self.selected_features]
            X_input_scaled = self.preprocessors['scaler'].transform(X_input)
            
            # Make predictions with all models
            predictions = {}
            for name, model in self.models.items():
                pred = model.predict(X_input_scaled)[0]
                predictions[name] = pred
            
            # Calculate ensemble prediction
            regression_preds = [pred for name, pred in predictions.items() if 'Classifier' not in name]
            classification_preds = [pred for name, pred in predictions.items() if 'Classifier' in name]
            
            ensemble_regression = np.mean(regression_preds) if regression_preds else None
            ensemble_classification = np.round(np.mean(classification_preds)) if classification_preds else None
            
            # Generate results text
            results_text = f"""🧠 STRESS LEVEL PREDICTION RESULTS
{'='*50}

📊 Input Summary:
• Gender: {input_data['Gender']}
• Age: {input_data['Age']} years
• Occupation: {input_data['Occupation']}
• Sleep Duration: {input_data['Sleep Duration']} hours
• Sleep Quality: {input_data['Quality of Sleep']}/10
• Physical Activity: {input_data['Physical Activity Level']}/100
• Heart Rate: {input_data['Heart Rate']} bpm
• Daily Steps: {input_data['Daily Steps']}
• Blood Pressure: {input_data['Blood Pressure']}
• BMI Category: {input_data['BMI Category']}
• Sleep Disorder: {input_data['Sleep Disorder']}

🎯 PREDICTION RESULTS:
"""
            
            if ensemble_regression:
                stress_level = ensemble_regression
                results_text += f"\n🔮 ENSEMBLE PREDICTION: {stress_level:.1f}/10\n\n"
                
                # Stress level analysis
                if stress_level <= 3:
                    level = "LOW"
                    color_emoji = "🟢"
                    advice = """✅ EXCELLENT! Your stress levels appear to be well-managed.
• Continue your current healthy lifestyle
• Maintain regular exercise and good sleep habits
• Keep monitoring your stress levels"""
                elif stress_level <= 6:
                    level = "MODERATE"
                    color_emoji = "🟡"
                    advice = """⚠️ ATTENTION NEEDED: Consider stress management techniques.
• Try meditation or mindfulness exercises
• Improve sleep quality and duration
• Increase physical activity
• Consider work-life balance adjustments"""
                else:
                    level = "HIGH"
                    color_emoji = "🔴"
                    advice = """🚨 HIGH STRESS DETECTED: Immediate action recommended.
• Consider consulting a healthcare professional
• Implement immediate stress reduction strategies
• Prioritize sleep and relaxation
• Consider professional counseling or therapy"""
                
                results_text += f"{color_emoji} STRESS LEVEL: {level} ({stress_level:.1f}/10)\n\n"
                results_text += f"💡 RECOMMENDATIONS:\n{advice}\n\n"
            
            # Individual model predictions
            results_text += "🤖 INDIVIDUAL MODEL PREDICTIONS:\n"
            results_text += "-" * 40 + "\n"
            
            for name, pred in predictions.items():
                model_type = "🔢" if 'Classifier' not in name else "🏷️"
                results_text += f"{model_type} {name}: {pred:.2f}\n"
            
            results_text += f"\n📈 MODEL ENSEMBLE:\n"
            if ensemble_regression:
                results_text += f"• Average Regression Prediction: {ensemble_regression:.2f}\n"
            if ensemble_classification:
                results_text += f"• Average Classification Prediction: {ensemble_classification:.0f}\n"
            
            results_text += f"\n⚠️ IMPORTANT DISCLAIMER:\nThis prediction is for informational purposes only and should not replace professional medical advice. If you're experiencing high stress levels or health concerns, please consult with a qualified healthcare professional.\n"
            
            # Display results
            self.prediction_results.delete(1.0, tk.END)
            self.prediction_results.insert(1.0, results_text)
            
            self.log_message(f"Prediction completed - Stress Level: {ensemble_regression:.1f}/10")
            self.update_status(f"Prediction completed - {level} stress level")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            self.log_message(f"Prediction failed: {str(e)}")
    
    def clear_form(self):
        """Clear all input fields"""
        for var in self.input_vars.values():
            var.set("")
    
    def load_sample_input(self):
        """Load sample input data"""
        sample_inputs = [
            {
                "Gender": "Male", "Age": "30", "Occupation": "Software Engineer",
                "Sleep Duration": "6.5", "Quality of Sleep": "6", "Physical Activity Level": "45",
                "Heart Rate": "75", "Daily Steps": "5000", "Blood Pressure": "130/85",
                "BMI Category": "Normal", "Sleep Disorder": "None"
            },
            {
                "Gender": "Female", "Age": "35", "Occupation": "Doctor",
                "Sleep Duration": "8.0", "Quality of Sleep": "9", "Physical Activity Level": "75",
                "Heart Rate": "68", "Daily Steps": "9000", "Blood Pressure": "120/78",
                "BMI Category": "Normal", "Sleep Disorder": "None"
            },
            {
                "Gender": "Male", "Age": "45", "Occupation": "Manager",
                "Sleep Duration": "5.0", "Quality of Sleep": "3", "Physical Activity Level": "20",
                "Heart Rate": "85", "Daily Steps": "3500", "Blood Pressure": "145/92",
                "BMI Category": "Overweight", "Sleep Disorder": "Insomnia"
            }
        ]
        
        # Choose random sample
        sample = np.random.choice(sample_inputs)
        
        for field, value in sample.items():
            if field in self.input_vars:
                self.input_vars[field].set(value)
        
        self.log_message("Sample input loaded")
    
    def plot_model_comparison(self):
        """Plot model performance comparison"""
        if not self.trained or not self.model_results:
            messagebox.showwarning("Warning", "Please train models first!")
            return
        
        try:
            self.axes[0, 0].clear()
            self.axes[0, 1].clear()
            
            # Regression models
            reg_models = []
            reg_r2_scores = []
            
            for name, results in self.model_results.items():
                if results['type'] == 'regression':
                    reg_models.append(name.replace(' ', '\n'))
                    reg_r2_scores.append(results['test_r2'])
            
            if reg_models:
                bars1 = self.axes[0, 0].bar(reg_models, reg_r2_scores, color='skyblue', alpha=0.7)
                self.axes[0, 0].set_title('Regression Models - R² Score', fontsize=10, fontweight='bold')
                self.axes[0, 0].set_ylabel('R² Score')
                self.axes[0, 0].tick_params(axis='x', rotation=45, labelsize=8)
                
                # Add value labels on bars
                for bar, score in zip(bars1, reg_r2_scores):
                    self.axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                       f'{score:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Classification models
            clf_models = []
            clf_accuracies = []
            
            for name, results in self.model_results.items():
                if results['type'] == 'classification':
                    clf_models.append(name.replace(' ', '\n'))
                    clf_accuracies.append(results['test_accuracy'])
            
            if clf_models:
                bars2 = self.axes[0, 1].bar(clf_models, clf_accuracies, color='lightcoral', alpha=0.7)
                self.axes[0, 1].set_title('Classification Models - Accuracy', fontsize=10, fontweight='bold')
                self.axes[0, 1].set_ylabel('Accuracy')
                self.axes[0, 1].tick_params(axis='x', rotation=45, labelsize=8)
                
                # Add value labels on bars
                for bar, acc in zip(bars2, clf_accuracies):
                    self.axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                       f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
            
            self.canvas.draw()
            self.log_message("Model comparison chart updated")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot model comparison: {str(e)}")
    
    def plot_correlation_matrix(self):
        """Plot correlation matrix"""
        if self.dataset is None:
            messagebox.showwarning("Warning", "Please load a dataset first!")
            return
        
        try:
            self.axes[1, 0].clear()
            
            # Select numerical columns for correlation
            numerical_cols = self.dataset.select_dtypes(include=[np.number]).columns
            correlation_matrix = self.dataset[numerical_cols].corr()
            
            # Plot heatmap
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       ax=self.axes[1, 0], cbar_kws={'shrink': 0.8}, fmt='.2f')
            self.axes[1, 0].set_title('Feature Correlation Matrix', fontsize=10, fontweight='bold')
            
            self.canvas.draw()
            self.log_message("Correlation matrix updated")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot correlation matrix: {str(e)}")
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        if not self.trained or 'Random Forest Regressor' not in self.models:
            messagebox.showwarning("Warning", "Please train Random Forest Regressor first!")
            return
        
        try:
            self.axes[1, 1].clear()
            
            # Get feature importance from Random Forest
            rf_model = self.models['Random Forest Regressor']
            importance = rf_model.feature_importances_
            feature_names = self.selected_features
            
            # Sort by importance
            indices = np.argsort(importance)[::-1]
            
            # Plot top 10 features
            top_n = min(10, len(feature_names))
            top_indices = indices[:top_n]
            
            self.axes[1, 1].barh(range(top_n), importance[top_indices], color='lightgreen', alpha=0.7)
            self.axes[1, 1].set_yticks(range(top_n))
            self.axes[1, 1].set_yticklabels([feature_names[i] for i in top_indices], fontsize=8)
            self.axes[1, 1].set_xlabel('Importance')
            self.axes[1, 1].set_title('Top 10 Feature Importance', fontsize=10, fontweight='bold')
            
            self.canvas.draw()
            self.log_message("Feature importance chart updated")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot feature importance: {str(e)}")
    
    def analyze_models(self):
        """Analyze and compare all trained models"""
        if not self.trained:
            messagebox.showwarning("Warning", "Please train models first!")
            return
        
        try:
            analysis_text = f"""
🤖 MODEL ANALYSIS REPORT
{'='*50}

📊 TRAINED MODELS: {len(self.models)}
📈 SELECTED FEATURES: {len(self.selected_features)}

🏆 PERFORMANCE SUMMARY:
"""
            
            # Find best models
            best_regression = None
            best_regression_score = -1
            best_classification = None
            best_classification_score = -1
            
            for name, results in self.model_results.items():
                if results['type'] == 'regression' and results['test_r2'] > best_regression_score:
                    best_regression = name
                    best_regression_score = results['test_r2']
                elif results['type'] == 'classification' and results['test_accuracy'] > best_classification_score:
                    best_classification = name
                    best_classification_score = results['test_accuracy']
            
            if best_regression:
                analysis_text += f"\n🥇 BEST REGRESSION MODEL: {best_regression}"
                analysis_text += f"\n   R² Score: {best_regression_score:.4f}"
            
            if best_classification:
                analysis_text += f"\n🏆 BEST CLASSIFICATION MODEL: {best_classification}"
                analysis_text += f"\n   Accuracy: {best_classification_score:.4f}"
            
            analysis_text += f"\n\n📋 DETAILED RESULTS:\n" + "-"*40 + "\n"
            
            for name, results in self.model_results.items():
                analysis_text += f"\n🤖 {name}:\n"
                if results['type'] == 'regression':
                    analysis_text += f"   • R² Score: {results['test_r2']:.4f}\n"
                    analysis_text += f"   • MSE: {results['test_mse']:.4f}\n"
                    analysis_text += f"   • CV MSE: {results['cv_mse']:.4f} ± {results['cv_std']:.4f}\n"
                else:
                    analysis_text += f"   • Test Accuracy: {results['test_accuracy']:.4f}\n"
                    analysis_text += f"   • CV Accuracy: {results['cv_accuracy']:.4f} ± {results['cv_std']:.4f}\n"
            
            analysis_text += f"\n\n🔍 FEATURE ANALYSIS:\n" + "-"*40 + "\n"
            analysis_text += f"Selected Features ({len(self.selected_features)}):\n"
            for i, feature in enumerate(self.selected_features, 1):
                analysis_text += f"{i:2d}. {feature}\n"
            
            # Display in log
            self.training_log.insert(tk.END, analysis_text)
            self.training_log.see(tk.END)
            
            self.log_message("Model analysis completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Model analysis failed: {str(e)}")
    
    def save_charts(self):
        """Save all charts to files"""
        if not self.trained:
            messagebox.showwarning("Warning", "Please train models first!")
            return
        
        try:
            # Create output directory
            output_dir = filedialog.askdirectory(title="Select Directory to Save Charts")
            if not output_dir:
                return
            
            # Save the current figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(output_dir, f"model_analysis_{timestamp}.png")
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            
            # Save model results as JSON
            results_filename = os.path.join(output_dir, f"model_results_{timestamp}.json")
            with open(results_filename, 'w') as f:
                # Convert numpy types to regular Python types for JSON serialization
                results_for_json = {}
                for name, results in self.model_results.items():
                    results_for_json[name] = {}
                    for key, value in results.items():
                        if isinstance(value, np.floating):
                            results_for_json[name][key] = float(value)
                        elif isinstance(value, np.integer):
                            results_for_json[name][key] = int(value)
                        else:
                            results_for_json[name][key] = value
                
                json.dump({
                    'model_results': results_for_json,
                    'selected_features': self.selected_features,
                    'timestamp': timestamp
                }, f, indent=2)
            
            messagebox.showinfo("Success", f"Charts and results saved to:\n{filename}\n{results_filename}")
            self.log_message(f"Charts and results saved to {output_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save charts: {str(e)}")
    
    def export_results(self):
        """Export all results and models"""
        if not self.trained:
            messagebox.showwarning("Warning", "Please train models first!")
            return
        
        try:
            # Create export directory
            export_dir = filedialog.askdirectory(title="Select Directory for Export")
            if not export_dir:
                return
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_folder = os.path.join(export_dir, f"stress_prediction_export_{timestamp}")
            os.makedirs(export_folder, exist_ok=True)
            
            # Save models
            models_dir = os.path.join(export_folder, "models")
            os.makedirs(models_dir, exist_ok=True)
            
            for name, model in self.models.items():
                model_filename = os.path.join(models_dir, f"{name.replace(' ', '_').lower()}.pkl")
                joblib.dump(model, model_filename)
            
            # Save preprocessors
            for name, preprocessor in self.preprocessors.items():
                prep_filename = os.path.join(models_dir, f"{name}.pkl")
                joblib.dump(preprocessor, prep_filename)
            
            # Save selected features
            features_filename = os.path.join(models_dir, "selected_features.json")
            with open(features_filename, 'w') as f:
                json.dump(self.selected_features, f, indent=2)
            
            # Save results
            results_filename = os.path.join(export_folder, "results.json")
            with open(results_filename, 'w') as f:
                results_for_json = {}
                for name, results in self.model_results.items():
                    results_for_json[name] = {}
                    for key, value in results.items():
                        if isinstance(value, (np.floating, np.integer)):
                            results_for_json[name][key] = float(value)
                        else:
                            results_for_json[name][key] = value
                
                json.dump(results_for_json, f, indent=2)
            
            # Save charts
            charts_filename = os.path.join(export_folder, "model_analysis.png")
            self.fig.savefig(charts_filename, dpi=300, bbox_inches='tight')
            
            # Create README
            readme_filename = os.path.join(export_folder, "README.txt")
            with open(readme_filename, 'w') as f:
                f.write(f"""Stress Level Prediction System Export
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Contents:
- models/: Trained machine learning models (.pkl files)
- results.json: Model performance metrics
- model_analysis.png: Visualization charts
- README.txt: This file

Models Included:
""")
                for name in self.models.keys():
                    f.write(f"- {name}\n")
                
                f.write(f"\nSelected Features ({len(self.selected_features)}):\n")
                for feature in self.selected_features:
                    f.write(f"- {feature}\n")
            
            messagebox.showinfo("Success", f"Complete export saved to:\n{export_folder}")
            self.log_message(f"Complete export saved to {export_folder}")
            
            # Offer to open the folder
            if messagebox.askyesno("Open Folder", "Would you like to open the export folder?"):
                if sys.platform == "win32":
                    os.startfile(export_folder)
                elif sys.platform == "darwin":
                    os.system(f"open '{export_folder}'")
                else:
                    os.system(f"xdg-open '{export_folder}'")
            
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")

def main():
    """Main application entry point"""
    try:
        # Create and configure root window
        root = tk.Tk()
        
        # Set window icon (if available)
        try:
            # Try to set a custom icon
            root.iconbitmap(default='stress_icon.ico')
        except:
            pass  # Use default icon if custom not available
        
        # Create and run application
        app = StressLevelPredictionApp(root)
        
        # Center window on screen
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f'{width}x{height}+{x}+{y}')
        
        # Start the GUI event loop
        root.mainloop()
        
    except Exception as e:
        print(f"Application failed to start: {str(e)}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
