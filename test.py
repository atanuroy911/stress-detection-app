import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ModelTester:
    def __init__(self, test_data_path=None):
        self.test_data_path = test_data_path or '../Datasets/Sleep_health_and_lifestyle_dataset.csv'
        self.models = {}
        self.encoders = {}
        self.scaler = None
        self.selected_features = None
        self.test_results = {}
        
    def load_models_and_preprocessors(self):
        """Load all trained models and preprocessors"""
        print("Loading models and preprocessors...")
        
        try:
            # Load preprocessors
            self.scaler = joblib.load('models/scaler.pkl')
            self.selected_features = joblib.load('models/selected_features.pkl')
            self.encoders['gender'] = joblib.load('models/gender_encoder.pkl')
            self.encoders['occupation'] = joblib.load('models/occupation_encoder.pkl')
            
            # Load regression models
            regression_models = [
                'linear_regression', 'ridge_regression', 'lasso_regression',
                'random_forest_regressor', 'gradient_boosting_regressor',
                'svr', 'mlp_regressor', 'decision_tree_regressor', 'knn_regressor'
            ]
            
            for model_name in regression_models:
                try:
                    model_path = f'models/{model_name}_regression.pkl'
                    self.models[f'{model_name}_regression'] = joblib.load(model_path)
                    print(f"Loaded {model_name}_regression")
                except FileNotFoundError:
                    print(f"Warning: {model_path} not found")
            
            # Load classification models
            classification_models = [
                'logistic_regression', 'random_forest_classifier', 'svc'
            ]
            
            for model_name in classification_models:
                try:
                    model_path = f'models/{model_name}_classification.pkl'
                    self.models[f'{model_name}_classification'] = joblib.load(model_path)
                    print(f"Loaded {model_name}_classification")
                except FileNotFoundError:
                    print(f"Warning: {model_path} not found")
            
            # Load unsupervised models
            unsupervised_models = ['kmeans_clustering', 'hierarchical_clustering', 'pca']
            
            for model_name in unsupervised_models:
                try:
                    model_path = f'models/{model_name}.pkl'
                    self.models[model_name] = joblib.load(model_path)
                    print(f"Loaded {model_name}")
                except FileNotFoundError:
                    print(f"Warning: {model_path} not found")
            
            print(f"Successfully loaded {len(self.models)} models")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
        
        return True
    
    def preprocess_test_data(self, df=None):
        """Preprocess test data using the same pipeline as training"""
        if df is None:
            print("Loading test data...")
            df = pd.read_csv(self.test_data_path)
        
        print("Preprocessing test data...")
        df_processed = df.copy()
        
        # Apply same feature engineering as training
        # Handle Blood Pressure
        df_processed[['Systolic_BP', 'Diastolic_BP']] = df_processed['Blood Pressure'].str.split('/', expand=True).astype(int)
        
        # Create new features
        df_processed['BMI_Risk_Score'] = df_processed['BMI Category'].map({
            'Normal': 0, 'Normal Weight': 0, 'Overweight': 1, 'Obese': 2
        })
        
        df_processed['Sleep_Quality_Score'] = df_processed['Quality of Sleep'] * df_processed['Sleep Duration']
        df_processed['Activity_Sleep_Ratio'] = df_processed['Physical Activity Level'] / df_processed['Sleep Duration']
        df_processed['Steps_per_Hour_Awake'] = df_processed['Daily Steps'] / (24 - df_processed['Sleep Duration'])
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
        
        # Encode categorical variables using fitted encoders
        df_processed['Gender_Encoded'] = self.encoders['gender'].transform(df_processed['Gender'])
        df_processed['Occupation_Encoded'] = self.encoders['occupation'].transform(df_processed['Occupation'])
        
        # Select features and target
        X = df_processed[self.selected_features]
        y = df_processed['Stress Level'] if 'Stress Level' in df_processed.columns else None
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        return X, X_scaled, y, df_processed
    
    def test_regression_models(self, X_scaled, y_true):
        """Test all regression models"""
        print("\nTesting Regression Models...")
        regression_results = {}
        
        for model_name, model in self.models.items():
            if 'regression' in model_name:
                print(f"Testing {model_name}...")
                
                # Make predictions
                y_pred = model.predict(X_scaled)
                
                # Calculate metrics
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_true, y_pred)
                mae = np.mean(np.abs(y_true - y_pred))
                
                regression_results[model_name] = {
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'mae': mae,
                    'predictions': y_pred.tolist()
                }
                
                print(f"  MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")
        
        return regression_results
    
    def test_classification_models(self, X_scaled, y_true):
        """Test all classification models"""
        print("\nTesting Classification Models...")
        classification_results = {}
        
        for model_name, model in self.models.items():
            if 'classification' in model_name:
                print(f"Testing {model_name}...")
                
                # Make predictions
                y_pred = model.predict(X_scaled)
                y_pred_proba = None
                
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_scaled)
                
                # Calculate metrics
                accuracy = accuracy_score(y_true, y_pred)
                
                # Classification report
                class_report = classification_report(y_true, y_pred, output_dict=True)
                
                # Confusion matrix
                conf_matrix = confusion_matrix(y_true, y_pred)
                
                classification_results[model_name] = {
                    'accuracy': accuracy,
                    'classification_report': class_report,
                    'confusion_matrix': conf_matrix.tolist(),
                    'predictions': y_pred.tolist(),
                    'prediction_probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None
                }
                
                print(f"  Accuracy: {accuracy:.4f}")
                
                # Plot confusion matrix
                plt.figure(figsize=(8, 6))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix - {model_name}')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                plt.savefig(f'static/confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        return classification_results
    
    def test_unsupervised_models(self, X_scaled):
        """Test unsupervised models"""
        print("\nTesting Unsupervised Models...")
        unsupervised_results = {}
        
        # Test clustering models
        for model_name, model in self.models.items():
            if 'clustering' in model_name:
                print(f"Testing {model_name}...")
                
                # Make predictions
                cluster_labels = model.fit_predict(X_scaled) if hasattr(model, 'fit_predict') else model.predict(X_scaled)
                
                unsupervised_results[model_name] = {
                    'cluster_labels': cluster_labels.tolist(),
                    'n_clusters': len(np.unique(cluster_labels))
                }
                
                print(f"  Number of clusters: {len(np.unique(cluster_labels))}")
        
        # Test PCA
        if 'pca' in self.models:
            print("Testing PCA...")
            pca_transformed = self.models['pca'].transform(X_scaled)
            
            unsupervised_results['pca'] = {
                'transformed_data': pca_transformed.tolist(),
                'n_components': pca_transformed.shape[1],
                'explained_variance_ratio': self.models['pca'].explained_variance_ratio_.tolist()
            }
            
            print(f"  Transformed shape: {pca_transformed.shape}")
        
        return unsupervised_results
    
    def test_single_prediction(self, input_data):
        """Test prediction for a single sample"""
        print("\nTesting Single Prediction...")
        
        # Create DataFrame from input
        if isinstance(input_data, dict):
            df_single = pd.DataFrame([input_data])
        else:
            df_single = pd.DataFrame(input_data)
        
        # Preprocess
        X, X_scaled, _, _ = self.preprocess_test_data(df_single)
        
        predictions = {}
        
        # Test regression models
        for model_name, model in self.models.items():
            if 'regression' in model_name:
                pred = model.predict(X_scaled)[0]
                predictions[f'{model_name}_stress_level'] = round(pred, 2)
        
        # Test classification models
        for model_name, model in self.models.items():
            if 'classification' in model_name:
                pred = model.predict(X_scaled)[0]
                predictions[f'{model_name}_stress_category'] = int(pred)
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_scaled)[0]
                    predictions[f'{model_name}_probabilities'] = proba.tolist()
        
        return predictions
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\nGenerating Test Report...")
        
        # Create comparison visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Regression models performance
        if self.test_results.get('regression'):
            reg_models = list(self.test_results['regression'].keys())
            reg_r2 = [self.test_results['regression'][model]['r2'] for model in reg_models]
            reg_rmse = [self.test_results['regression'][model]['rmse'] for model in reg_models]
            
            axes[0, 0].barh(reg_models, reg_r2)
            axes[0, 0].set_title('Regression Models - R² Score')
            axes[0, 0].set_xlabel('R² Score')
            
            axes[0, 1].barh(reg_models, reg_rmse)
            axes[0, 1].set_title('Regression Models - RMSE')
            axes[0, 1].set_xlabel('RMSE')
        
        # Classification models performance
        if self.test_results.get('classification'):
            clf_models = list(self.test_results['classification'].keys())
            clf_accuracy = [self.test_results['classification'][model]['accuracy'] for model in clf_models]
            
            axes[1, 0].bar(clf_models, clf_accuracy)
            axes[1, 0].set_title('Classification Models - Accuracy')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Model predictions vs actual (for best regression model)
        if self.test_results.get('regression'):
            best_reg_model = max(self.test_results['regression'], 
                               key=lambda x: self.test_results['regression'][x]['r2'])
            y_pred = self.test_results['regression'][best_reg_model]['predictions']
            
            axes[1, 1].scatter(self.y_true, y_pred, alpha=0.6)
            axes[1, 1].plot([min(self.y_true), max(self.y_true)], 
                          [min(self.y_true), max(self.y_true)], 'r--', lw=2)
            axes[1, 1].set_xlabel('Actual Stress Level')
            axes[1, 1].set_ylabel('Predicted Stress Level')
            axes[1, 1].set_title(f'Actual vs Predicted - {best_reg_model}')
        
        plt.tight_layout()
        plt.savefig('static/test_results_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save test results
        with open('models/test_results.json', 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*60)
        print("MODEL TESTING SUMMARY")
        print("="*60)
        
        if self.test_results.get('regression'):
            print("\nREGRESSION MODELS PERFORMANCE:")
            best_reg_r2 = 0
            best_reg_model = ""
            for model, metrics in self.test_results['regression'].items():
                model_name = model.replace('_regression', '').replace('_', ' ').title()
                print(f"{model_name:25} | R²: {metrics['r2']:.4f} | RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f}")
                if metrics['r2'] > best_reg_r2:
                    best_reg_r2 = metrics['r2']
                    best_reg_model = model_name
            print(f"\nBest Regression Model: {best_reg_model} (R²: {best_reg_r2:.4f})")
        
        if self.test_results.get('classification'):
            print("\nCLASSIFICATION MODELS PERFORMANCE:")
            best_clf_acc = 0
            best_clf_model = ""
            for model, metrics in self.test_results['classification'].items():
                model_name = model.replace('_classification', '').replace('_', ' ').title()
                print(f"{model_name:25} | Accuracy: {metrics['accuracy']:.4f}")
                if metrics['accuracy'] > best_clf_acc:
                    best_clf_acc = metrics['accuracy']
                    best_clf_model = model_name
            print(f"\nBest Classification Model: {best_clf_model} (Accuracy: {best_clf_acc:.4f})")
        
        if self.test_results.get('unsupervised'):
            print("\nUNSUPERVISED MODELS:")
            for model, metrics in self.test_results['unsupervised'].items():
                model_name = model.replace('_', ' ').title()
                if 'n_clusters' in metrics:
                    print(f"{model_name:25} | Clusters: {metrics['n_clusters']}")
                elif 'n_components' in metrics:
                    print(f"{model_name:25} | Components: {metrics['n_components']}")
    
    def run_comprehensive_test(self):
        """Run comprehensive testing of all models"""
        print("Starting Comprehensive Model Testing...")
        print("="*50)
        
        # Load models and preprocessors
        if not self.load_models_and_preprocessors():
            print("Failed to load models. Please run train.py first.")
            return False
        
        # Preprocess test data
        X, X_scaled, y_true, df_processed = self.preprocess_test_data()
        self.y_true = y_true
        
        if y_true is None:
            print("Warning: No target variable found in test data. Limited testing available.")
            return False
        
        # Test all model types
        self.test_results['regression'] = self.test_regression_models(X_scaled, y_true)
        self.test_results['classification'] = self.test_classification_models(X_scaled, y_true)
        self.test_results['unsupervised'] = self.test_unsupervised_models(X_scaled)
        
        # Generate report
        self.generate_test_report()
        
        print("\nTesting completed successfully!")
        print("Results saved to 'models/test_results.json'")
        print("Visualizations saved to 'static/' directory")
        
        return True

def test_sample_predictions():
    """Test predictions on sample data"""
    print("\nTesting Sample Predictions...")
    
    tester = ModelTester()
    if not tester.load_models_and_preprocessors():
        print("Failed to load models for sample testing.")
        return
    
    # Sample test cases
    sample_data = [
        {
            'Person ID': 999,
            'Gender': 'Male',
            'Age': 30,
            'Occupation': 'Software Engineer',
            'Sleep Duration': 6.5,
            'Quality of Sleep': 6,
            'Physical Activity Level': 45,
            'Heart Rate': 75,
            'Daily Steps': 5000,
            'Blood Pressure': '130/85',
            'BMI Category': 'Normal',
            'Sleep Disorder': 'None'
        },
        {
            'Person ID': 1000,
            'Gender': 'Female',
            'Age': 35,
            'Occupation': 'Doctor',
            'Sleep Duration': 7.0,
            'Quality of Sleep': 8,
            'Physical Activity Level': 60,
            'Heart Rate': 70,
            'Daily Steps': 8000,
            'Blood Pressure': '120/80',
            'BMI Category': 'Normal',
            'Sleep Disorder': 'None'
        }
    ]
    
    for i, sample in enumerate(sample_data):
        print(f"\nSample {i+1} Predictions:")
        predictions = tester.test_single_prediction(sample)
        for model, pred in predictions.items():
            print(f"  {model}: {pred}")

def main():
    # Run comprehensive testing
    tester = ModelTester()
    success = tester.run_comprehensive_test()
    
    if success:
        # Test sample predictions
        test_sample_predictions()
    
    print("\nAll testing completed!")

if __name__ == "__main__":
    main()
