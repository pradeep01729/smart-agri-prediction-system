"""
Prediction Pages Blueprint
Handles GET requests for the prediction forms and server-rendered POST fallbacks.
"""

from datetime import datetime
import json
from flask import Blueprint, current_app, render_template, request, flash, redirect, url_for
import numpy as np
import pandas as pd
from app.models import PredictionLog, db

predict_bp = Blueprint('predict', __name__, url_prefix='/predict')

@predict_bp.route('/yield', methods=['GET', 'POST'])
def predict_yield_view():
    if request.method == 'GET':
        return render_template('predict_yield.html')

    # Handle standard POST form submission fallback
    form_data = request.form.to_dict()
    models = current_app.config.get('ML_MODELS', {})
    model_bundle = models.get('yield')
    
    if not model_bundle:
        flash("Machine learning model is currently unavailable. Please train models first.", "danger")
        return render_template('predict_yield.html', form_data=form_data)

    crop_type = form_data.get('crop_type', '').strip().lower()
    valid_crops = ['wheat', 'rice', 'maize', 'cotton', 'sugarcane']
    if crop_type not in valid_crops:
        flash(f"Please select a valid crop type from: {', '.join(valid_crops)}", "warning")
        return render_template('predict_yield.html', form_data=form_data)

    required_fields = ['soil_ph', 'rainfall_mm', 'temperature_c', 'humidity_pct', 'nitrogen', 'phosphorus', 'potassium']
    clean_inputs = {'crop_type': crop_type}
    
    for f in required_fields:
        val = form_data.get(f)
        if val is None or str(val).strip() == '':
            flash(f"Field '{f}' is required.", "warning")
            return render_template('predict_yield.html', form_data=form_data)
        try:
            clean_inputs[f] = float(val)
        except ValueError:
            flash(f"Field '{f}' must be a valid number.", "warning")
            return render_template('predict_yield.html', form_data=form_data)

    try:
        preprocessor = model_bundle['preprocessor']
        rf = model_bundle['estimator']

        X_df = pd.DataFrame([clean_inputs])
        X_trans = preprocessor.transform(X_df)

        tree_preds = np.array([tree.predict(X_trans)[0] for tree in rf.estimators_])
        mean_pred = float(np.mean(tree_preds))
        std_pred = float(np.std(tree_preds))
        conf_margin = float(1.96 * std_pred)
        
        rel_std = std_pred / max(mean_pred, 1.0)
        confidence_pct = max(55.0, min(98.5, 100.0 - (rel_std * 140.0)))
        
        lower_bound = max(0.0, mean_pred - conf_margin)
        upper_bound = mean_pred + conf_margin
        formatted_result = f"{mean_pred:.2f} kg/ha"

        log = PredictionLog(
            prediction_type='yield',
            inputs_json=json.dumps(clean_inputs),
            result=formatted_result,
            confidence=round(confidence_pct, 1)
        )
        db.session.add(log)
        db.session.commit()

        result_data = {
            'predicted_yield': round(mean_pred, 2),
            'confidence_pct': round(confidence_pct, 1),
            'confidence_interval': [round(lower_bound, 2), round(upper_bound, 2)],
            'std_deviation': round(std_pred, 2),
            'crop_type': crop_type,
            'unit': 'kg/ha',
            'inputs': clean_inputs
        }
        return render_template('predict_yield.html', result=result_data, form_data=form_data)
    except Exception as e:
        db.session.rollback()
        flash(f"Error computing yield prediction: {str(e)}", "danger")
        return render_template('predict_yield.html', form_data=form_data)

@predict_bp.route('/disease', methods=['GET', 'POST'])
def predict_disease_view():
    if request.method == 'GET':
        return render_template('predict_disease.html')

    form_data = request.form.to_dict()
    models = current_app.config.get('ML_MODELS', {})
    model_bundle = models.get('disease')
    
    if not model_bundle:
        flash("Disease risk model is currently unavailable.", "danger")
        return render_template('predict_disease.html', form_data=form_data)

    crop_type = form_data.get('crop_type', '').strip().lower()
    valid_crops = ['wheat', 'rice', 'maize', 'cotton', 'sugarcane']
    if crop_type not in valid_crops:
        flash("Invalid crop selected.", "warning")
        return render_template('predict_disease.html', form_data=form_data)

    clean_inputs = {'crop_type': crop_type}
    num_fields = ['temperature_c', 'humidity_pct', 'rainfall_mm', 'leaf_wetness_hours', 'prior_disease_history']
    for f in num_fields:
        val = form_data.get(f)
        if val is None or str(val).strip() == '':
            flash(f"Field '{f}' is required.", "warning")
            return render_template('predict_disease.html', form_data=form_data)
        try:
            clean_inputs[f] = int(val) if f == 'prior_disease_history' else float(val)
        except ValueError:
            flash(f"Field '{f}' must be a number.", "warning")
            return render_template('predict_disease.html', form_data=form_data)

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

        log = PredictionLog(
            prediction_type='disease',
            inputs_json=json.dumps(clean_inputs),
            result=risk_level.capitalize(),
            confidence=round(confidence_pct, 1)
        )
        db.session.add(log)
        db.session.commit()

        advisories = {
            'low': 'Environmental conditions are unfavorable for pathogen infection. Continue standard monitoring.',
            'medium': 'Moderate risk detected. Increase field scouting and ensure soil drainage.',
            'high': 'CRITICAL: High disease outbreak probability. Urgent preventative fungicide or protective measures advised.'
        }

        result_data = {
            'risk_level': risk_level,
            'confidence_pct': round(confidence_pct, 1),
            'class_probabilities': prob_breakdown,
            'advisory': advisories.get(risk_level, ''),
            'crop_type': crop_type,
            'inputs': clean_inputs
        }
        return render_template('predict_disease.html', result=result_data, form_data=form_data)
    except Exception as e:
        db.session.rollback()
        flash(f"Error calculating disease risk: {str(e)}", "danger")
        return render_template('predict_disease.html', form_data=form_data)

@predict_bp.route('/fertilizer', methods=['GET', 'POST'])
def predict_fertilizer_view():
    if request.method == 'GET':
        return render_template('predict_fertilizer.html')

    form_data = request.form.to_dict()
    models = current_app.config.get('ML_MODELS', {})
    model_bundle = models.get('fertilizer')
    
    if not model_bundle:
        flash("Fertilizer model is currently unavailable.", "danger")
        return render_template('predict_fertilizer.html', form_data=form_data)

    crop_type = form_data.get('crop_type', '').strip().lower()
    valid_crops = ['wheat', 'rice', 'maize', 'cotton', 'sugarcane']
    if crop_type not in valid_crops:
        flash("Invalid crop selected.", "warning")
        return render_template('predict_fertilizer.html', form_data=form_data)

    clean_inputs = {'crop_type': crop_type}
    num_fields = ['soil_ph', 'nitrogen', 'phosphorus', 'potassium', 'moisture_pct']
    for f in num_fields:
        val = form_data.get(f)
        if val is None or str(val).strip() == '':
            flash(f"Field '{f}' is required.", "warning")
            return render_template('predict_fertilizer.html', form_data=form_data)
        try:
            clean_inputs[f] = float(val)
        except ValueError:
            flash(f"Field '{f}' must be a number.", "warning")
            return render_template('predict_fertilizer.html', form_data=form_data)

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

        fertilizer_details = {
            'Urea': 'High Nitrogen (46-0-0) source for rapid vegetative growth and leaf greenness.',
            'DAP': 'Diammonium Phosphate (18-46-0) to stimulate root branching and early seedling vigor.',
            'MOP': 'Muriate of Potash (0-0-60) to strengthen stems and enhance drought and pest resilience.',
            'NPK': 'Balanced fertilizer formulation for total nutrient replenishment and steady development.',
            'Compost': 'Organic matter supplement to regulate soil pH buffering and enhance moisture holding capacity.'
        }

        log = PredictionLog(
            prediction_type='fertilizer',
            inputs_json=json.dumps(clean_inputs),
            result=recommended_fertilizer,
            confidence=round(confidence_pct, 1)
        )
        db.session.add(log)
        db.session.commit()

        result_data = {
            'recommended_fertilizer': recommended_fertilizer,
            'confidence_pct': round(confidence_pct, 1),
            'class_probabilities': prob_breakdown,
            'fertilizer_info': fertilizer_details.get(recommended_fertilizer, ''),
            'crop_type': crop_type,
            'inputs': clean_inputs
        }
        return render_template('predict_fertilizer.html', result=result_data, form_data=form_data)
    except Exception as e:
        db.session.rollback()
        flash(f"Error computing fertilizer recommendation: {str(e)}", "danger")
        return render_template('predict_fertilizer.html', form_data=form_data)
