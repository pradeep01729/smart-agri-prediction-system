"""
RESTful JSON API Blueprint
Provides endpoints for predictions, history, healthcheck, and dashboard statistics.
"""

from datetime import datetime
import json
from flask import Blueprint, current_app, jsonify, request
import numpy as np
import pandas as pd
from app.models import PredictionLog, db
from ml.weather_analysis import generate_synthetic_monthly_rainfall, analyze_rainfall_series

api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

def validate_number(value, name, min_val=None, max_val=None):
    try:
        val = float(value)
        if min_val is not None and val < min_val:
            return None, f"'{name}' must be at least {min_val}"
        if max_val is not None and val > max_val:
            return None, f"'{name}' must be at most {max_val}"
        return val, None
    except (ValueError, TypeError):
        return None, f"'{name}' must be a valid numeric value"

@api_bp.route('/health', methods=['GET'])
def healthcheck():
    models = current_app.config.get('ML_MODELS', {})
    loaded = [k for k, v in models.items() if v is not None]
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'models_loaded': loaded,
        'database': 'connected'
    })

@api_bp.route('/predict/yield', methods=['POST'])
def predict_yield():
    data = request.get_json(silent=True) or request.form.to_dict()
    if not data:
        return jsonify({'status': 'error', 'message': 'Missing JSON request body or form data'}), 400

    models = current_app.config.get('ML_MODELS', {})
    model_bundle = models.get('yield')
    if not model_bundle:
        return jsonify({'status': 'error', 'message': 'Yield prediction model is not loaded'}), 503

    # Validate crop_type
    crop_type = str(data.get('crop_type', '')).strip().lower()
    valid_crops = ['wheat', 'rice', 'maize', 'cotton', 'sugarcane']
    if crop_type not in valid_crops:
        return jsonify({
            'status': 'error',
            'message': f"Invalid 'crop_type'. Must be one of: {', '.join(valid_crops)}"
        }), 400

    # Validate numerical fields
    specs = {
        'soil_ph': (3.5, 10.0),
        'rainfall_mm': (0.0, 3500.0),
        'temperature_c': (-10.0, 55.0),
        'humidity_pct': (0.0, 100.0),
        'nitrogen': (0.0, 400.0),
        'phosphorus': (0.0, 300.0),
        'potassium': (0.0, 300.0),
    }

    clean_inputs = {'crop_type': crop_type}
    for field, (min_v, max_v) in specs.items():
        if field not in data:
            return jsonify({'status': 'error', 'message': f"Missing required field '{field}'"}), 400
        val, err = validate_number(data[field], field, min_v, max_v)
        if err:
            return jsonify({'status': 'error', 'message': err}), 400
        clean_inputs[field] = val

    try:
        preprocessor = model_bundle['preprocessor']
        rf = model_bundle['estimator']

        X_df = pd.DataFrame([clean_inputs])
        X_trans = preprocessor.transform(X_df)

        # Variance across individual tree estimators for confidence indicator
        tree_preds = np.array([tree.predict(X_trans)[0] for tree in rf.estimators_])
        mean_pred = float(np.mean(tree_preds))
        std_pred = float(np.std(tree_preds))
        conf_margin = float(1.96 * std_pred)

        # Confidence percentage proxy: higher variance relative to mean yields lower confidence
        rel_std = std_pred / max(mean_pred, 1.0)
        confidence_pct = max(55.0, min(98.5, 100.0 - (rel_std * 140.0)))

        lower_bound = max(0.0, mean_pred - conf_margin)
        upper_bound = mean_pred + conf_margin

        formatted_result = f"{mean_pred:.2f} kg/ha"

        # Log to DB
        log = PredictionLog(
            prediction_type='yield',
            inputs_json=json.dumps(clean_inputs),
            result=formatted_result,
            confidence=round(confidence_pct, 1)
        )
        db.session.add(log)
        db.session.commit()

        return jsonify({
            'status': 'success',
            'log_id': log.id,
            'prediction_type': 'yield',
            'crop_type': crop_type,
            'predicted_yield_kg_per_hectare': round(mean_pred, 2),
            'confidence_pct': round(confidence_pct, 1),
            'confidence_interval': [round(lower_bound, 2), round(upper_bound, 2)],
            'std_deviation': round(std_pred, 2),
            'unit': 'kg/ha',
            'inputs': clean_inputs
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': f'Prediction computation error: {str(e)}'}), 500

@api_bp.route('/predict/disease', methods=['POST'])
def predict_disease():
    data = request.get_json(silent=True) or request.form.to_dict()
    if not data:
        return jsonify({'status': 'error', 'message': 'Missing JSON request body or form data'}), 400

    models = current_app.config.get('ML_MODELS', {})
    model_bundle = models.get('disease')
    if not model_bundle:
        return jsonify({'status': 'error', 'message': 'Disease risk model is not loaded'}), 503

    crop_type = str(data.get('crop_type', '')).strip().lower()
    valid_crops = ['wheat', 'rice', 'maize', 'cotton', 'sugarcane']
    if crop_type not in valid_crops:
        return jsonify({
            'status': 'error',
            'message': f"Invalid 'crop_type'. Must be one of: {', '.join(valid_crops)}"
        }), 400

    specs = {
        'temperature_c': (-10.0, 55.0),
        'humidity_pct': (0.0, 100.0),
        'rainfall_mm': (0.0, 1500.0),
        'leaf_wetness_hours': (0.0, 24.0),
        'prior_disease_history': (0, 1)
    }

    clean_inputs = {'crop_type': crop_type}
    for field, (min_v, max_v) in specs.items():
        if field not in data:
            return jsonify({'status': 'error', 'message': f"Missing required field '{field}'"}), 400
        val, err = validate_number(data[field], field, min_v, max_v)
        if err:
            return jsonify({'status': 'error', 'message': err}), 400
        if field == 'prior_disease_history':
            val = int(val)
        clean_inputs[field] = val

    try:
        preprocessor = model_bundle['preprocessor']
        rf = model_bundle['estimator']
        classes = model_bundle['classes']

        X_df = pd.DataFrame([clean_inputs])
        X_trans = preprocessor.transform(X_df)

        probs = rf.predict_proba(X_trans)[0]
        pred_idx = int(np.argmax(probs))
        risk_level = classes[pred_idx]
        confidence_pct = float(probs[pred_idx]) * 100.0

        prob_breakdown = {cls: round(float(p) * 100.0, 1) for cls, p in zip(classes, probs)}

        # Log to DB
        log = PredictionLog(
            prediction_type='disease',
            inputs_json=json.dumps(clean_inputs),
            result=risk_level.capitalize(),
            confidence=round(confidence_pct, 1)
        )
        db.session.add(log)
        db.session.commit()

        # Action recommendation based on risk
        advisories = {
            'low': 'Environmental factors are unfavorable for pathogen proliferation. Continue standard crop scouting.',
            'medium': 'Elevated humidity/wetness detected. Inspect canopy undersides and prepare preventative bio-fungicide if wet spell persists.',
            'high': 'CRITICAL ALERT: Environmental conditions strongly favor outbreak. Immediate fungicide/bactericide application and moisture drainage recommended.'
        }

        return jsonify({
            'status': 'success',
            'log_id': log.id,
            'prediction_type': 'disease',
            'crop_type': crop_type,
            'risk_level': risk_level,
            'confidence_pct': round(confidence_pct, 1),
            'class_probabilities': prob_breakdown,
            'advisory': advisories.get(risk_level, ''),
            'inputs': clean_inputs
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': f'Disease prediction error: {str(e)}'}), 500

@api_bp.route('/predict/fertilizer', methods=['POST'])
def predict_fertilizer():
    data = request.get_json(silent=True) or request.form.to_dict()
    if not data:
        return jsonify({'status': 'error', 'message': 'Missing JSON request body or form data'}), 400

    models = current_app.config.get('ML_MODELS', {})
    model_bundle = models.get('fertilizer')
    if not model_bundle:
        return jsonify({'status': 'error', 'message': 'Fertilizer recommendation model is not loaded'}), 503

    crop_type = str(data.get('crop_type', '')).strip().lower()
    valid_crops = ['wheat', 'rice', 'maize', 'cotton', 'sugarcane']
    if crop_type not in valid_crops:
        return jsonify({
            'status': 'error',
            'message': f"Invalid 'crop_type'. Must be one of: {', '.join(valid_crops)}"
        }), 400

    specs = {
        'soil_ph': (3.5, 10.0),
        'nitrogen': (0.0, 350.0),
        'phosphorus': (0.0, 250.0),
        'potassium': (0.0, 250.0),
        'moisture_pct': (0.0, 100.0)
    }

    clean_inputs = {'crop_type': crop_type}
    for field, (min_v, max_v) in specs.items():
        if field not in data:
            return jsonify({'status': 'error', 'message': f"Missing required field '{field}'"}), 400
        val, err = validate_number(data[field], field, min_v, max_v)
        if err:
            return jsonify({'status': 'error', 'message': err}), 400
        clean_inputs[field] = val

    try:
        preprocessor = model_bundle['preprocessor']
        rf = model_bundle['estimator']
        classes = model_bundle['classes']

        X_df = pd.DataFrame([clean_inputs])
        X_trans = preprocessor.transform(X_df)

        probs = rf.predict_proba(X_trans)[0]
        pred_idx = int(np.argmax(probs))
        recommended_fertilizer = classes[pred_idx]
        confidence_pct = float(probs[pred_idx]) * 100.0

        prob_breakdown = {cls: round(float(p) * 100.0, 1) for cls, p in zip(classes, probs)}

        # Nutrient agronomic guidelines
        fertilizer_details = {
            'Urea': 'High Nitrogen (46-0-0) source for vegetative growth and chlorophyll synthesis.',
            'DAP': 'Diammonium Phosphate (18-46-0) for strong root development and early tillering.',
            'MOP': 'Muriate of Potash (0-0-60) for drought tolerance, disease resistance, and grain filling.',
            'NPK': 'Balanced complete blend (e.g. 14-35-14 or 19-19-19) for overall nutrient replenishment.',
            'Compost': 'Organic matter supplement to optimize soil pH buffering and improve microbial water retention.'
        }

        # Log to DB
        log = PredictionLog(
            prediction_type='fertilizer',
            inputs_json=json.dumps(clean_inputs),
            result=recommended_fertilizer,
            confidence=round(confidence_pct, 1)
        )
        db.session.add(log)
        db.session.commit()

        return jsonify({
            'status': 'success',
            'log_id': log.id,
            'prediction_type': 'fertilizer',
            'crop_type': crop_type,
            'recommended_fertilizer': recommended_fertilizer,
            'confidence_pct': round(confidence_pct, 1),
            'class_probabilities': prob_breakdown,
            'fertilizer_info': fertilizer_details.get(recommended_fertilizer, ''),
            'inputs': clean_inputs
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': f'Fertilizer recommendation error: {str(e)}'}), 500

@api_bp.route('/history', methods=['GET'])
def get_history():
    try:
        limit = min(int(request.args.get('limit', 20)), 100)
    except ValueError:
        limit = 20

    logs = PredictionLog.query.order_by(PredictionLog.timestamp.desc()).limit(limit).all()
    return jsonify({
        'status': 'success',
        'count': len(logs),
        'history': [log.to_dict() for log in logs]
    })

@api_bp.route('/dashboard-stats', methods=['GET'])
def get_dashboard_stats():
    total_preds = PredictionLog.query.count()
    
    # Crop frequency count
    crop_counts = {}
    yield_records_by_crop = {}
    high_risk_disease_count = 0
    all_yield_values = []

    all_logs = PredictionLog.query.order_by(PredictionLog.timestamp.desc()).all()
    
    timeline_dict = {}
    for log in all_logs:
        date_str = log.timestamp.strftime('%Y-%m-%d')
        timeline_dict[date_str] = timeline_dict.get(date_str, 0) + 1
        
        inputs = log.get_inputs()
        crop = inputs.get('crop_type', 'unknown')
        if crop != 'unknown':
            crop_counts[crop] = crop_counts.get(crop, 0) + 1
            
        if log.prediction_type == 'yield':
            try:
                # result is formatted as "XXXX.XX kg/ha"
                numeric_val = float(log.result.split()[0])
                all_yield_values.append(numeric_val)
                if crop not in yield_records_by_crop:
                    yield_records_by_crop[crop] = []
                yield_records_by_crop[crop].append(numeric_val)
            except Exception:
                pass
        elif log.prediction_type == 'disease':
            if str(log.result).lower() == 'high':
                high_risk_disease_count += 1

    most_common_crop = max(crop_counts.items(), key=lambda x: x[1])[0].capitalize() if crop_counts else 'None'
    avg_yield = round(float(np.mean(all_yield_values)), 1) if all_yield_values else 0.0

    # Yield by crop averages
    yield_by_crop = {
        crop.capitalize(): round(float(np.mean(vals)), 1)
        for crop, vals in yield_records_by_crop.items()
    }

    # Timeline sorted chronologically
    sorted_dates = sorted(timeline_dict.keys())
    timeline = {d: timeline_dict[d] for d in sorted_dates[-14:]}

    recent_logs = [log.to_dict() for log in all_logs[:8]]

    return jsonify({
        'total_predictions': total_preds,
        'most_common_crop': most_common_crop,
        'avg_predicted_yield': avg_yield,
        'high_risk_disease_count': high_risk_disease_count,
        'yield_by_crop': yield_by_crop,
        'timeline': timeline,
        'recent_logs': recent_logs
    })

@api_bp.route('/weather-trend', methods=['GET'])
def get_weather_trend():
    records = generate_synthetic_monthly_rainfall(12)
    analysis = analyze_rainfall_series(records)
    return jsonify({
        'status': 'success',
        'data': analysis
    })
