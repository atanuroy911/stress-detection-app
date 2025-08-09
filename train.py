import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy import stats
import joblib
import warnings
warnings.filterwarnings('ignore')

class StressLevelPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_and_explore_data(self):
        """Load and explore the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        
        print("\nDataset Info:")
        print(f"Shape: {self.df.shape}")
        print(f"\nColumns: {list(self.df.columns)}")
        print(f"\nData types:\n{self.df.dtypes}")
        print(f"\nMissing values:\n{self.df.isnull().sum()}")
        print(f"\nBasic statistics:\n{self.df.describe()}")
        
        # Check unique values for categorical variables
        categorical_cols = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
        for col in categorical_cols:
            if col in self.df.columns:
                print(f"\n{col} unique values: {self.df[col].unique()}")
        
        # Target variable distribution
        print(f"\nStress Level distribution:\n{self.df['Stress Level'].value_counts().sort_index()}")
        
        return self.df
    
    def feature_engineering(self):
        """Comprehensive feature engineering with correlation analysis"""
        print("\nPerforming Feature Engineering...")
        
        # Create a copy for processing
        df_processed = self.df.copy()
        
        # Handle Blood Pressure - split into systolic and diastolic
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
                                                   labels=[0, 1, 2])  # Low, Normal, High
        
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
        le_gender = LabelEncoder()
        df_processed['Gender_Encoded'] = le_gender.fit_transform(df_processed['Gender'])
        
        le_occupation = LabelEncoder()
        df_processed['Occupation_Encoded'] = le_occupation.fit_transform(df_processed['Occupation'])
        
        # Save encoders
        joblib.dump(le_gender, 'models/gender_encoder.pkl')
        joblib.dump(le_occupation, 'models/occupation_encoder.pkl')
        
        # Select numerical features for correlation analysis
        numerical_features = [
            'Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
            'Heart Rate', 'Daily Steps', 'Systolic_BP', 'Diastolic_BP',
            'BMI_Risk_Score', 'Sleep_Quality_Score', 'Activity_Sleep_Ratio',
            'Steps_per_Hour_Awake', 'BP_Risk', 'Heart_Rate_Category',
            'Has_Sleep_Disorder', 'Sleep_Disorder_Type', 'Age_Group',
            'Gender_Encoded', 'Occupation_Encoded'
        ]
        
        # Create correlation matrix
        correlation_matrix = df_processed[numerical_features + ['Stress Level']].corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(15, 12))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('static/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature selection based on correlation with target
        target_correlation = correlation_matrix['Stress Level'].abs().sort_values(ascending=False)
        print("\nFeature correlation with Stress Level:")
        print(target_correlation)
        
        # Select features with correlation > 0.1 (excluding target itself)
        selected_features = target_correlation[target_correlation > 0.1].index.tolist()
        selected_features.remove('Stress Level')
        
        print(f"\nSelected features based on correlation: {selected_features}")
        
        # Prepare final feature matrix
        self.X = df_processed[selected_features]
        self.y = df_processed['Stress Level']
        
        # Save feature names
        joblib.dump(selected_features, 'models/selected_features.pkl')
        
        print(f"\nFinal feature matrix shape: {self.X.shape}")
        
        return self.X, self.y
    
    def split_and_scale_data(self):
        """Split and scale the data"""
        print("\nSplitting and scaling data...")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Save scaler
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_supervised_models(self):
        """Train various supervised learning models"""
        print("\nTraining Supervised Learning Models...")
        
        # Regression models (treating stress as continuous)
        regression_models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0),
            'MLP Regressor': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
            'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
            'KNN Regressor': KNeighborsRegressor(n_neighbors=5)
        }
        
        # Classification models (treating stress as categorical)
        classification_models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVC': SVC(kernel='rbf', C=1.0, random_state=42)
        }
        
        regression_results = {}
        classification_results = {}
        
        # Train regression models
        print("\nTraining Regression Models:")
        for name, model in regression_models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Predictions
            y_pred_train = model.predict(self.X_train_scaled)
            y_pred_test = model.predict(self.X_test_scaled)
            
            # Metrics
            train_mse = mean_squared_error(self.y_train, y_pred_train)
            test_mse = mean_squared_error(self.y_test, y_pred_test)
            train_r2 = r2_score(self.y_train, y_pred_train)
            test_r2 = r2_score(self.y_test, y_pred_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                      cv=5, scoring='neg_mean_squared_error')
            
            regression_results[name] = {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'cv_mse': -cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            # Save model
            joblib.dump(model, f'models/{name.lower().replace(" ", "_")}_regression.pkl')
        
        # Train classification models
        print("\nTraining Classification Models:")
        for name, model in classification_models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Predictions
            y_pred_train = model.predict(self.X_train_scaled)
            y_pred_test = model.predict(self.X_test_scaled)
            
            # Metrics
            train_acc = accuracy_score(self.y_train, y_pred_train)
            test_acc = accuracy_score(self.y_test, y_pred_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                      cv=5, scoring='accuracy')
            
            classification_results[name] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'cv_accuracy': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            # Save model
            joblib.dump(model, f'models/{name.lower().replace(" ", "_")}_classification.pkl')
        
        self.results['regression'] = regression_results
        self.results['classification'] = classification_results
        
        return regression_results, classification_results
    
    def train_unsupervised_models(self):
        """Train unsupervised learning models"""
        print("\nTraining Unsupervised Learning Models...")
        
        unsupervised_results = {}
        
        # K-Means Clustering
        print("Training K-Means...")
        # Determine optimal number of clusters using elbow method
        inertias = []
        K_range = range(2, 11)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.X_train_scaled)
            inertias.append(kmeans.inertia_)
        
        # Plot elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(K_range, inertias, 'bo-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        plt.grid(True)
        plt.savefig('static/kmeans_elbow.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Train K-Means with optimal k (let's use k=3 for stress levels: low, medium, high)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(self.X_train_scaled)
        
        # Analyze clusters vs actual stress levels
        cluster_stress_analysis = pd.DataFrame({
            'Cluster': cluster_labels,
            'Actual_Stress': self.y_train.values
        })
        
        cluster_summary = cluster_stress_analysis.groupby('Cluster')['Actual_Stress'].agg(['mean', 'std', 'count'])
        print("\nK-Means Cluster Analysis:")
        print(cluster_summary)
        
        unsupervised_results['KMeans'] = {
            'n_clusters': 3,
            'inertia': kmeans.inertia_,
            'cluster_summary': cluster_summary.to_dict()
        }
        
        joblib.dump(kmeans, 'models/kmeans_clustering.pkl')
        
        # Hierarchical Clustering
        print("Training Hierarchical Clustering...")
        hierarchical = AgglomerativeClustering(n_clusters=3)
        hierarchical_labels = hierarchical.fit_predict(self.X_train_scaled)
        
        hierarchical_analysis = pd.DataFrame({
            'Cluster': hierarchical_labels,
            'Actual_Stress': self.y_train.values
        })
        
        hierarchical_summary = hierarchical_analysis.groupby('Cluster')['Actual_Stress'].agg(['mean', 'std', 'count'])
        print("\nHierarchical Cluster Analysis:")
        print(hierarchical_summary)
        
        unsupervised_results['Hierarchical'] = {
            'n_clusters': 3,
            'cluster_summary': hierarchical_summary.to_dict()
        }
        
        joblib.dump(hierarchical, 'models/hierarchical_clustering.pkl')
        
        # PCA for dimensionality reduction
        print("Applying PCA...")
        pca = PCA()
        pca.fit(self.X_train_scaled)
        
        # Explained variance ratio
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        
        # Plot PCA
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'bo-')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA - Explained Variance Ratio')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'ro-')
        plt.axhline(y=0.95, color='k', linestyle='--', alpha=0.7)
        plt.xlabel('Principal Component')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('PCA - Cumulative Explained Variance')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('static/pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Find number of components for 95% variance
        n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
        
        unsupervised_results['PCA'] = {
            'total_components': len(explained_variance_ratio),
            'components_for_95_variance': n_components_95,
            'explained_variance_ratio': explained_variance_ratio.tolist()
        }
        
        joblib.dump(pca, 'models/pca.pkl')
        
        self.results['unsupervised'] = unsupervised_results
        return unsupervised_results
    
    def generate_report(self):
        """Generate comprehensive model comparison report"""
        print("\nGenerating Model Comparison Report...")
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Regression models comparison
        reg_models = list(self.results['regression'].keys())
        reg_test_mse = [self.results['regression'][model]['test_mse'] for model in reg_models]
        reg_test_r2 = [self.results['regression'][model]['test_r2'] for model in reg_models]
        
        axes[0, 0].bar(reg_models, reg_test_mse)
        axes[0, 0].set_title('Regression Models - Test MSE')
        axes[0, 0].set_ylabel('Mean Squared Error')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        axes[0, 1].bar(reg_models, reg_test_r2)
        axes[0, 1].set_title('Regression Models - Test R²')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Classification models comparison
        clf_models = list(self.results['classification'].keys())
        clf_test_acc = [self.results['classification'][model]['test_accuracy'] for model in clf_models]
        clf_cv_acc = [self.results['classification'][model]['cv_accuracy'] for model in clf_models]
        
        x_pos = np.arange(len(clf_models))
        width = 0.35
        
        axes[1, 0].bar(x_pos - width/2, clf_test_acc, width, label='Test Accuracy')
        axes[1, 0].bar(x_pos + width/2, clf_cv_acc, width, label='CV Accuracy')
        axes[1, 0].set_title('Classification Models - Accuracy Comparison')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(clf_models, rotation=45)
        axes[1, 0].legend()
        
        # Feature importance (using Random Forest)
        rf_reg = joblib.load('models/random_forest_regressor_regression.pkl')
        feature_names = joblib.load('models/selected_features.pkl')
        feature_importance = rf_reg.feature_importances_
        
        sorted_idx = np.argsort(feature_importance)[-10:]  # Top 10 features
        axes[1, 1].barh(np.array(feature_names)[sorted_idx], feature_importance[sorted_idx])
        axes[1, 1].set_title('Top 10 Feature Importance (Random Forest)')
        axes[1, 1].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('static/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed results
        results_df = pd.DataFrame({
            'Model_Type': ['Regression'] * len(reg_models) + ['Classification'] * len(clf_models),
            'Model_Name': reg_models + clf_models,
            'Primary_Metric': reg_test_r2 + clf_test_acc,
            'Secondary_Metric': reg_test_mse + clf_cv_acc
        })
        
        results_df.to_csv('static/model_results.csv', index=False)
        
        # Print summary
        print("\n" + "="*50)
        print("MODEL COMPARISON SUMMARY")
        print("="*50)
        
        print("\nREGRESSION MODELS:")
        best_reg_model = max(self.results['regression'], 
                           key=lambda x: self.results['regression'][x]['test_r2'])
        for model, metrics in self.results['regression'].items():
            print(f"{model:25} | R²: {metrics['test_r2']:.4f} | MSE: {metrics['test_mse']:.4f}")
        print(f"\nBest Regression Model: {best_reg_model}")
        
        print("\nCLASSIFICATION MODELS:")
        best_clf_model = max(self.results['classification'], 
                           key=lambda x: self.results['classification'][x]['test_accuracy'])
        for model, metrics in self.results['classification'].items():
            print(f"{model:25} | Accuracy: {metrics['test_accuracy']:.4f} | CV: {metrics['cv_accuracy']:.4f}")
        print(f"\nBest Classification Model: {best_clf_model}")
        
        print("\nUNSUPERVISED LEARNING:")
        for model, metrics in self.results['unsupervised'].items():
            if model == 'PCA':
                print(f"{model:25} | Components for 95% var: {metrics['components_for_95_variance']}")
            else:
                print(f"{model:25} | Clusters: {metrics['n_clusters']}")
        
        # Save results to JSON for web app
        import json
        with open('models/results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        return self.results

def main():
    # Initialize the predictor
    predictor = StressLevelPredictor('../Datasets/Sleep_health_and_lifestyle_dataset.csv')
    
    # Load and explore data
    predictor.load_and_explore_data()
    
    # Feature engineering
    predictor.feature_engineering()
    
    # Split and scale data
    predictor.split_and_scale_data()
    
    # Train supervised models
    predictor.train_supervised_models()
    
    # Train unsupervised models
    predictor.train_unsupervised_models()
    
    # Generate report
    predictor.generate_report()
    
    print("\nTraining completed! All models saved in 'models/' directory.")
    print("Results and visualizations saved in 'static/' directory.")

if __name__ == "__main__":
    main()
