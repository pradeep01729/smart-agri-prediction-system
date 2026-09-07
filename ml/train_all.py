import sys
import os
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from ml.train_yield_model import train_yield_model
    from ml.train_disease_model import train_disease_model
    from ml.train_fertilizer_model import train_fertilizer_model
except ImportError:
    from train_yield_model import train_yield_model
    from train_disease_model import train_disease_model
    from train_fertilizer_model import train_fertilizer_model

def train_all():
    print("\n" + "#"*60)
    print("STARTING SMART AGRI MACHINE LEARNING MODEL TRAINING PIPELINE")
    print("#"*60 + "\n")
    
    start_time = time.time()
    
    print(">>> STEP 1/3: Training Crop Yield Regression Model...")
    yield_bundle = train_yield_model()
    
    print("\n>>> STEP 2/3: Training Disease Risk Classification Model...")
    disease_bundle = train_disease_model()
    
    print("\n>>> STEP 3/3: Training Fertilizer Recommendation Model...")
    fertilizer_bundle = train_fertilizer_model()
    
    total_time = time.time() - start_time
    print("\n" + "#"*60)
    print(f"ALL MODELS TRAINED AND SAVED SUCCESSFULLY IN {total_time:.2f}s!")
    print(f"Yield Model:      R² = {yield_bundle['metrics']['r2']}, RMSE = {yield_bundle['metrics']['rmse']} kg/ha")
    print(f"Disease Model:    Accuracy = {disease_bundle['metrics']['accuracy'] * 100:.2f}%")
    print(f"Fertilizer Model: Accuracy = {fertilizer_bundle['metrics']['accuracy'] * 100:.2f}%")
    print("#"*60 + "\n")

if __name__ == '__main__':
    train_all()
