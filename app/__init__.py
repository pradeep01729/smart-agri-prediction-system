"""
Flask Application Factory
Initializes SQLAlchemy, registers blueprints, and pre-loads ML models at startup.
"""

import os
import joblib
from flask import Flask
from app.models import db

def load_ml_models(base_dir: str):
    models = {}
    models_dir = os.path.join(base_dir, 'ml', 'models')
    
    yield_path = os.path.join(models_dir, 'yield_model.pkl')
    disease_path = os.path.join(models_dir, 'disease_model.pkl')
    fertilizer_path = os.path.join(models_dir, 'fertilizer_model.pkl')
    
    if os.path.exists(yield_path):
        models['yield'] = joblib.load(yield_path)
    else:
        print(f"[Warning] Yield model not found at {yield_path}")
        
    if os.path.exists(disease_path):
        models['disease'] = joblib.load(disease_path)
    else:
        print(f"[Warning] Disease model not found at {disease_path}")
        
    if os.path.exists(fertilizer_path):
        models['fertilizer'] = joblib.load(fertilizer_path)
    else:
        print(f"[Warning] Fertilizer model not found at {fertilizer_path}")
        
    return models

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    
    # Configure database and secret key
    os.makedirs(app.instance_path, exist_ok=True)
    db_path = os.path.join(app.instance_path, 'agri_system.db')
    
    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'agri-secret-key-2026'),
        SQLALCHEMY_DATABASE_URI=f'sqlite:///{db_path}',
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )
    
    if test_config:
        app.config.update(test_config)
        
    db.init_app(app)
    
    # Pre-load ML models once at app startup
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    app.config['ML_MODELS'] = load_ml_models(base_dir)
    app.config['PROJECT_ROOT'] = base_dir
    
    with app.app_context():
        db.create_all()
        
    # Register blueprints
    from app.routes.dashboard import dashboard_bp
    from app.routes.predict import predict_bp
    from app.routes.api import api_bp
    
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(api_bp)
    
    return app
