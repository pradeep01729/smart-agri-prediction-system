"""
Dashboard & Documentation Routes
Renders the primary analytics dashboard and API documentation page.
"""

from flask import Blueprint, render_template
import numpy as np
from app.models import PredictionLog

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/', methods=['GET'])
def index():
    # Fetch initial server-side summary values for immediate display
    total_preds = PredictionLog.query.count()
    all_logs = PredictionLog.query.order_by(PredictionLog.timestamp.desc()).all()
    
    crop_counts = {}
    yield_values = []
    high_risk_count = 0
    
    for log in all_logs:
        inputs = log.get_inputs()
        crop = inputs.get('crop_type', '')
        if crop:
            crop_counts[crop] = crop_counts.get(crop, 0) + 1
            
        if log.prediction_type == 'yield':
            try:
                val = float(log.result.split()[0])
                yield_values.append(val)
            except Exception:
                pass
        elif log.prediction_type == 'disease':
            if str(log.result).lower() == 'high':
                high_risk_count += 1

    most_common_crop = max(crop_counts.items(), key=lambda x: x[1])[0].capitalize() if crop_counts else 'N/A'
    avg_yield = f"{np.mean(yield_values):.1f} kg/ha" if yield_values else "N/A"

    recent_logs = all_logs[:8]

    return render_template(
        'dashboard.html',
        total_predictions=total_preds,
        most_common_crop=most_common_crop,
        avg_predicted_yield=avg_yield,
        high_risk_disease_count=high_risk_count,
        recent_logs=recent_logs
    )

@dashboard_bp.route('/api-docs', methods=['GET'])
def api_docs():
    return render_template('api_docs.html')
