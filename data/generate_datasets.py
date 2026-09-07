"""
Synthetic Agricultural Dataset Generator
Generates realistic agricultural datasets with domain-grounded correlations:
1. crop_yield.csv: Multi-feature yield regression
2. disease_risk.csv: Environmental conditions to disease risk classification (low/medium/high)
3. fertilizer.csv: Soil and nutrient analysis to fertilizer recommendation
"""

import os
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd

def ensure_dir(dir_path: str):
    os.makedirs(dir_path, exist_ok=True)

def generate_crop_yield_data(n_samples: int = 1200, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    crops = ['wheat', 'rice', 'maize', 'cotton', 'sugarcane']
    
    # Baseline yield distributions in kg/hectare
    crop_yield_bases = {
        'wheat': 3800,
        'rice': 4600,
        'maize': 5500,
        'cotton': 2400,
        'sugarcane': 72000
    }
    
    # Ideal temperature (°C) and rainfall (mm) ranges per crop
    crop_profiles = {
        'wheat': {'temp_opt': 20.0, 'temp_std': 4.0, 'rain_opt': 550.0, 'rain_std': 120.0},
        'rice': {'temp_opt': 29.0, 'temp_std': 3.5, 'rain_opt': 1250.0, 'rain_std': 250.0},
        'maize': {'temp_opt': 26.0, 'temp_std': 4.0, 'rain_opt': 750.0, 'rain_std': 150.0},
        'cotton': {'temp_opt': 28.0, 'temp_std': 4.5, 'rain_opt': 680.0, 'rain_std': 140.0},
        'sugarcane': {'temp_opt': 31.0, 'temp_std': 4.0, 'rain_opt': 1450.0, 'rain_std': 280.0},
    }
    
    data = []
    for _ in range(n_samples):
        crop = np.random.choice(crops)
        profile = crop_profiles[crop]
        
        # Environmental factors
        soil_ph = float(np.clip(np.random.normal(6.6, 0.7), 4.5, 9.0))
        rainfall_mm = float(np.clip(np.random.normal(profile['rain_opt'], profile['rain_std']), 200.0, 2200.0))
        temperature_c = float(np.clip(np.random.normal(profile['temp_opt'], profile['temp_std']), 10.0, 44.0))
        humidity_pct = float(np.clip(np.random.normal(68.0, 14.0), 30.0, 98.0))
        
        # Soil macronutrients (kg/ha equivalent test index)
        nitrogen = float(np.clip(np.random.normal(110.0, 35.0), 20.0, 220.0))
        phosphorus = float(np.clip(np.random.normal(45.0, 18.0), 10.0, 100.0))
        potassium = float(np.clip(np.random.normal(55.0, 20.0), 15.0, 120.0))
        
        # Multi-factor yield multiplier calculation
        base_yield = crop_yield_bases[crop]
        
        # pH factor: optimal between 6.2 and 7.3, penalized quadratically outside
        ph_penalty = 1.0 - 0.12 * ((soil_ph - 6.75) ** 2)
        ph_factor = float(np.clip(ph_penalty, 0.55, 1.05))
        
        # Weather factor: distance from optimal temperature and rainfall
        temp_dist = abs(temperature_c - profile['temp_opt']) / 10.0
        temp_factor = float(np.clip(1.0 - 0.15 * temp_dist, 0.6, 1.05))
        
        rain_ratio = rainfall_mm / profile['rain_opt']
        rain_factor = float(np.clip(1.0 - 0.35 * ((rain_ratio - 1.0) ** 2), 0.5, 1.1))
        
        # Nutrient response (saturating log curves)
        n_factor = float(np.clip(0.6 + 0.45 * (nitrogen / 140.0), 0.5, 1.15))
        p_factor = float(np.clip(0.7 + 0.35 * (phosphorus / 60.0), 0.6, 1.1))
        k_factor = float(np.clip(0.75 + 0.30 * (potassium / 70.0), 0.65, 1.1))
        
        # Stochastic residual variation (±7%)
        noise = np.random.normal(1.0, 0.07)
        
        final_yield = base_yield * ph_factor * temp_factor * rain_factor * n_factor * p_factor * k_factor * noise
        final_yield = float(np.round(np.maximum(final_yield, base_yield * 0.35), 2))
        
        data.append({
            'crop_type': crop,
            'soil_ph': round(soil_ph, 2),
            'rainfall_mm': round(rainfall_mm, 1),
            'temperature_c': round(temperature_c, 1),
            'humidity_pct': round(humidity_pct, 1),
            'nitrogen': round(nitrogen, 1),
            'phosphorus': round(phosphorus, 1),
            'potassium': round(potassium, 1),
            'yield_kg_per_hectare': final_yield
        })
        
    return pd.DataFrame(data)

def generate_disease_risk_data(n_samples: int = 1200, seed: int = 43) -> pd.DataFrame:
    np.random.seed(seed)
    crops = ['wheat', 'rice', 'maize', 'cotton', 'sugarcane']
    
    data = []
    for _ in range(n_samples):
        crop = np.random.choice(crops)
        temperature_c = float(np.clip(np.random.normal(27.0, 6.0), 12.0, 42.0))
        humidity_pct = float(np.clip(np.random.normal(72.0, 16.0), 35.0, 99.0))
        rainfall_mm = float(np.clip(np.random.exponential(120.0) + 20.0, 10.0, 600.0))
        leaf_wetness_hours = float(np.clip(np.random.normal(humidity_pct / 9.0, 3.0), 0.0, 24.0))
        prior_disease_history = int(np.random.choice([0, 1], p=[0.65, 0.35]))
        
        # Risk score calculation
        # Fungal & bacterial pathogens thrive with prolonged leaf wetness, high humidity (>78%), warm temps (22-32°C)
        wetness_score = (leaf_wetness_hours / 24.0) * 35.0
        humidity_score = ((humidity_pct - 50.0) / 50.0) * 30.0 if humidity_pct > 50 else 0.0
        temp_score = 20.0 - abs(temperature_c - 27.0) * 1.5
        temp_score = max(0.0, temp_score)
        history_score = 15.0 if prior_disease_history == 1 else 0.0
        rain_score = min(15.0, (rainfall_mm / 300.0) * 15.0)
        
        total_score = wetness_score + humidity_score + temp_score + history_score + rain_score + np.random.normal(0, 4.0)
        
        if total_score < 42.0:
            risk_level = 'low'
        elif total_score < 68.0:
            risk_level = 'medium'
        else:
            risk_level = 'high'
            
        data.append({
            'crop_type': crop,
            'temperature_c': round(temperature_c, 1),
            'humidity_pct': round(humidity_pct, 1),
            'rainfall_mm': round(rainfall_mm, 1),
            'leaf_wetness_hours': round(leaf_wetness_hours, 1),
            'prior_disease_history': prior_disease_history,
            'risk_level': risk_level
        })
        
    return pd.DataFrame(data)

def generate_fertilizer_data(n_samples: int = 1200, seed: int = 44) -> pd.DataFrame:
    np.random.seed(seed)
    crops = ['wheat', 'rice', 'maize', 'cotton', 'sugarcane']
    
    data = []
    for _ in range(n_samples):
        crop = np.random.choice(crops)
        soil_ph = float(np.clip(np.random.normal(6.5, 0.8), 4.6, 8.9))
        nitrogen = float(np.clip(np.random.normal(90.0, 45.0), 10.0, 200.0))
        phosphorus = float(np.clip(np.random.normal(40.0, 22.0), 5.0, 100.0))
        potassium = float(np.clip(np.random.normal(48.0, 24.0), 8.0, 110.0))
        moisture_pct = float(np.clip(np.random.normal(52.0, 18.0), 15.0, 95.0))
        
        # Agronomic decision heuristics with realistic noise
        # 1. Acidic/alkaline soils or very low organic moisture -> Compost
        if soil_ph < 5.6 or soil_ph > 8.0 or moisture_pct < 25.0:
            rec = 'Compost'
        # 2. Severe Nitrogen deficiency
        elif nitrogen < 55.0 and phosphorus >= 30.0 and potassium >= 35.0:
            rec = 'Urea'
        # 3. Severe Phosphorus deficiency
        elif phosphorus < 25.0 and potassium >= 35.0:
            rec = 'DAP'
        # 4. Severe Potassium deficiency
        elif potassium < 30.0 and nitrogen >= 50.0:
            rec = 'MOP'
        # 5. General / Balanced deficit or crop specific high requirement
        else:
            rec = 'NPK'
            
        # Add 5% natural variance/field misclassification
        if np.random.rand() < 0.05:
            rec = np.random.choice(['Urea', 'DAP', 'MOP', 'NPK', 'Compost'])
            
        data.append({
            'crop_type': crop,
            'soil_ph': round(soil_ph, 2),
            'nitrogen': round(nitrogen, 1),
            'phosphorus': round(phosphorus, 1),
            'potassium': round(potassium, 1),
            'moisture_pct': round(moisture_pct, 1),
            'recommended_fertilizer': rec
        })
        
    return pd.DataFrame(data)

def main():
    output_dir = os.path.join(os.path.dirname(__file__))
    ensure_dir(output_dir)
    
    yield_df = generate_crop_yield_data(1500)
    yield_path = os.path.join(output_dir, 'crop_yield.csv')
    yield_df.to_csv(yield_path, index=False)
    print(f"Generated crop_yield.csv with {len(yield_df)} rows at {yield_path}")
    
    disease_df = generate_disease_risk_data(1400)
    disease_path = os.path.join(output_dir, 'disease_risk.csv')
    disease_df.to_csv(disease_path, index=False)
    print(f"Generated disease_risk.csv with {len(disease_df)} rows at {disease_path}")
    
    fertilizer_df = generate_fertilizer_data(1400)
    fertilizer_path = os.path.join(output_dir, 'fertilizer.csv')
    fertilizer_df.to_csv(fertilizer_path, index=False)
    print(f"Generated fertilizer.csv with {len(fertilizer_df)} rows at {fertilizer_path}")

if __name__ == '__main__':
    main()
