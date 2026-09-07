"""
Crop Yield Model Training
Trains a RandomForestRegressor on crop_yield.csv with Pipeline preprocessing.
Saves model to ml/models/yield_model.pkl and feature importances to ml/models/yield_feature_importance.png.
"""

import os
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def train_yield_model():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(base_dir, 'data', 'crop_yield.csv')
    models_dir = os.path.join(base_dir, 'ml', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run generate_datasets.py first.")
        
    print(f"[Yield Model] Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    cat_features = ['crop_type']
    num_features = [
        'soil_ph', 'rainfall_mm', 'temperature_c',
        'humidity_pct', 'nitrogen', 'phosphorus', 'potassium'
    ]
    target = 'yield_kg_per_hectare'
    
    X = df[cat_features + num_features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
            ('num', StandardScaler(), num_features)
        ]
    )
    
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=14,
        min_samples_split=4,
        random_state=42,
        n_jobs=1
    )
    
    # Fit preprocessor & transform
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    rf.fit(X_train_trans, y_train)
    
    # Evaluation
    y_pred = rf.predict(X_test_trans)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print("\n" + "="*50)
    print("Crop Yield Model Evaluation")
    print("="*50)
    print(f"R² Score:              {r2:.4f}")
    print(f"Root Mean Sq Error:    {rmse:.2f} kg/ha")
    print(f"Mean Absolute Error:   {mae:.2f} kg/ha")
    print("="*50)
    
    # Feature importances extraction
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_encoded_names = cat_encoder.get_feature_names_out(cat_features).tolist()
    feature_names = cat_encoded_names + num_features
    importances = rf.feature_importances_
    
    # Plot feature importances
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(len(sorted_features)), sorted_importances[::-1], color='#2d7f45', align='center')
    plt.yticks(range(len(sorted_features)), sorted_features[::-1], fontsize=10)
    plt.xlabel('Random Forest Feature Importance', fontsize=11, fontweight='bold')
    plt.title('Crop Yield Prediction - Feature Importances', fontsize=13, fontweight='bold', pad=15)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plot_path = os.path.join(models_dir, 'yield_feature_importance.png')
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved feature importance plot to {plot_path}")
    
    # Package model bundle containing both preprocessor, rf estimator, and metadata
    model_bundle = {
        'preprocessor': preprocessor,
        'estimator': rf,
        'cat_features': cat_features,
        'num_features': num_features,
        'feature_names': feature_names,
        'metrics': {
            'r2': round(float(r2), 4),
            'rmse': round(float(rmse), 2),
            'mae': round(float(mae), 2)
        }
    }
    
    model_path = os.path.join(models_dir, 'yield_model.pkl')
    joblib.dump(model_bundle, model_path)
    print(f"Saved model bundle to {model_path}\n")
    return model_bundle

if __name__ == '__main__':
    train_yield_model()
