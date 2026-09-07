"""
Database Models for Smart Agricultural Prediction System
Uses SQLAlchemy to track and persist prediction logs and input features.
"""

from datetime import datetime
import json
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class PredictionLog(db.Model):
    __tablename__ = 'prediction_logs'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    prediction_type = db.Column(db.String(50), nullable=False, index=True)  # 'yield', 'disease', 'fertilizer'
    inputs_json = db.Column(db.Text, nullable=False)
    result = db.Column(db.String(200), nullable=False)
    confidence = db.Column(db.Float, nullable=True)  # Percentage or confidence proxy
    
    def get_inputs(self):
        try:
            return json.loads(self.inputs_json)
        except Exception:
            return {}
            
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'prediction_type': self.prediction_type,
            'inputs': self.get_inputs(),
            'result': self.result,
            'confidence': self.confidence
        }
