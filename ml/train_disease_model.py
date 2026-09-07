"""
Crop Disease Risk Model Training
Trains a RandomForestClassifier on disease_risk.csv.
Saves model to ml/models/disease_model.pkl and feature importances to ml/models/disease_feature_importance.png.
"""

import os
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def train_disease_model():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(base_dir, 'data', 'disease_risk.csv')
    models_dir = os.path.join(base_dir, 'ml', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run generate_datasets.py first.")
        
    print(f"[Disease Model] Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    cat_features = ['crop_type']
    num_features = [
        'temperature_c', 'humidity_pct', 'rainfall_mm',
        'leaf_wetness_hours', 'prior_disease_history'
    ]
    target = 'risk_level'
    
    X = df[cat_features + num_features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
            ('num', StandardScaler(), num_features)
        ]
    )
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=4,
        random_state=42,
        n_jobs=1
    )
    
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    rf.fit(X_train_trans, y_train)
    
    # Evaluation
    y_pred = rf.predict(X_test_trans)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=['low', 'medium', 'high'])
    report = classification_report(y_test, y_pred)
    
    print("\n" + "="*50)
    print("Crop Disease Risk Classification Evaluation")
    print("="*50)
    print(f"Accuracy: {acc * 100:.2f}%\n")
    print("Confusion Matrix (low, medium, high):")
    print(cm)
    print("\nClassification Report:")
    print(report)
    print("="*50)
    
    # Feature importances extraction
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_encoded_names = cat_encoder.get_feature_names_out(cat_features).tolist()
    feature_names = cat_encoded_names + num_features
    importances = rf.feature_importances_
    
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_features)), sorted_importances[::-1], color='#c2410c', align='center')
    plt.yticks(range(len(sorted_features)), sorted_features[::-1], fontsize=10)
    plt.xlabel('Random Forest Feature Importance', fontsize=11, fontweight='bold')
    plt.title('Disease Risk Prediction - Feature Importances', fontsize=13, fontweight='bold', pad=15)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plot_path = os.path.join(models_dir, 'disease_feature_importance.png')
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved feature importance plot to {plot_path}")
    
    model_bundle = {
        'preprocessor': preprocessor,
        'estimator': rf,
        'classes': rf.classes_.tolist(),
        'cat_features': cat_features,
        'num_features': num_features,
        'feature_names': feature_names,
        'metrics': {
            'accuracy': round(float(acc), 4)
        }
    }
    
    model_path = os.path.join(models_dir, 'disease_model.pkl')
    joblib.dump(model_bundle, model_path)
    print(f"Saved model bundle to {model_path}\n")
    return model_bundle

if __name__ == '__main__':
    train_disease_model()
