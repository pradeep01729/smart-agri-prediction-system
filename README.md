# Smart Agricultural Prediction System

A full-stack, end-to-end Machine Learning web application designed for final-year Computer Science, IT, and Agricultural Engineering capstone projects. The system delivers crop yield estimation, foliar pathogen/disease risk diagnosis, intelligent fertilizer recommendation, and precipitation trend forecasting.

---

## System Architecture & Features

- **Crop Yield Estimation:** Random Forest Regressor trained on soil pH, precipitation, ambient temperature, humidity, and macronutrient indices (N, P, K). Computes confidence intervals via variance across ensemble decision trees.
- **Disease Vulnerability Diagnosis:** Multi-class Random Forest Classifier categorizing pathogen outbreak risks (`low`, `medium`, `high`) based on sustained leaf wetness hours, atmospheric moisture, and field infection history.
- **Fertilizer Recommendation Engine:** Prescribes optimal soil amendments (`Urea`, `DAP`, `MOP`, `NPK`, `Compost`) mapped against nutrient deficiencies and soil pH buffering needs.
- **Seasonal Precipitation Analysis:** Time-series moving average smoothing and naive next-period projection for irrigation planning.
- **Interactive UI Dashboard:** Built with Bootstrap 5, semantic HTML5, Google Fonts, and Chart.js client-side visualizations. Forms support instant AJAX fetch submissions with inline results and one-click demo presets.
- **RESTful JSON API:** 5 core endpoints with OpenAPI-compatible payloads for programmatic access.
- **Persistent Telemetry:** SQLite database via Flask-SQLAlchemy tracking all model inferences and input snapshots.

---

## Project Structure

```
smart-agri-prediction-system/
├── app/
│   ├── __init__.py               # Flask application factory (loads ML models once at startup)
│   ├── models.py                 # SQLAlchemy PredictionLog schema
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── dashboard.py          # Dashboard view & API docs routes
│   │   ├── predict.py            # Yield, Disease, and Fertilizer form views & POST handlers
│   │   └── api.py                # RESTful JSON endpoints (predict, history, stats, health)
│   ├── static/
│   │   ├── css/
│   │   │   └── custom.css        # Modern, responsive agricultural theme
│   │   └── js/
│   │       ├── dashboard.js      # Chart.js initialization & dynamic telemetry
│   │       └── predict.js        # Client validation, fetch handlers, and presets
│   └── templates/
│       ├── base.html             # Shared navbar, alerts, footer layout
│       ├── dashboard.html        # 4 KPI cards + 2 interactive Chart.js charts
│       ├── predict_yield.html    # Yield form with inline confidence gauge
│       ├── predict_disease.html  # Disease form with probability distribution & advisory
│       ├── predict_fertilizer.html # Fertilizer form with agronomic rationale
│       └── api_docs.html         # Interactive API documentation with cURL examples
├── data/
│   ├── generate_datasets.py      # Synthetic agricultural dataset generator
│   ├── crop_yield.csv            # Generated dataset (~1500 rows)
│   ├── disease_risk.csv          # Generated dataset (~1400 rows)
│   └── fertilizer.csv            # Generated dataset (~1400 rows)
├── docs/
│   └── PROJECT_REPORT_OUTLINE.md # Academic report outline (Abstract to Future Scope)
├── ml/
│   ├── __init__.py
│   ├── train_yield_model.py      # RandomForestRegressor training script
│   ├── train_disease_model.py    # RandomForestClassifier disease training script
│   ├── train_fertilizer_model.py # RandomForestClassifier fertilizer training script
│   ├── train_all.py              # Master training orchestrator
│   ├── weather_analysis.py       # Moving-average trend & regression forecaster
│   └── models/                   # Serialized .pkl binaries & feature importance plots
│       ├── yield_model.pkl
│       ├── disease_model.pkl
│       ├── fertilizer_model.pkl
│       ├── yield_feature_importance.png
│       ├── disease_feature_importance.png
│       └── fertilizer_feature_importance.png
├── requirements.txt              # Pinned dependencies
├── run.py                        # Alternative entrypoint
└── README.md
```

---

## Quick Start & Setup

### Prerequisites
- Python 3.11+
- Git

### 1. Create Virtual Environment & Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate Synthetic Datasets
```bash
python data/generate_datasets.py
```
*Outputs `crop_yield.csv`, `disease_risk.csv`, and `fertilizer.csv` with realistic agronomic correlations.*

### 3. Train Machine Learning Models
```bash
python ml/train_all.py
```
*Trains all three models, reports evaluation metrics (R², accuracy, confusion matrix), and exports joblib `.pkl` artifacts and `.png` feature importance charts to `ml/models/`.*

### 4. Run the Web Application
```bash
flask --app app run --debug
```
*(Or alternatively run `python run.py`)*

Open your browser and navigate to: **`http://127.0.0.1:5000/`**

---

## REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Healthcheck and active model status |
| `POST` | `/api/v1/predict/yield` | Computes yield estimation & confidence interval |
| `POST` | `/api/v1/predict/disease` | Diagnoses disease vulnerability & advisory |
| `POST` | `/api/v1/predict/fertilizer` | Recommends fertilizer formulation |
| `GET` | `/api/v1/history` | Chronological log of recent inference queries |
| `GET` | `/api/v1/dashboard-stats` | Aggregated analytics & timeline for Chart.js |
| `GET` | `/api/v1/weather-trend` | Moving average and linear precipitation forecast |

### Example cURL Request:
```bash
curl -X POST http://127.0.0.1:5000/api/v1/predict/yield \
  -H "Content-Type: application/json" \
  -d '{
    "crop_type": "wheat",
    "soil_ph": 6.5,
    "rainfall_mm": 650.0,
    "temperature_c": 22.0,
    "humidity_pct": 60.0,
    "nitrogen": 120.0,
    "phosphorus": 45.0,
    "potassium": 40.0
  }'
```

---

## Known Limitations

> [!WARNING]
> **Synthetic Datasets & Academic Scope:**
> 1. **Not Agronomically Validated:** The datasets generated in `data/generate_datasets.py` are mathematically synthesized with realistic non-linear curves for demonstration and academic evaluation. They should **not** be used for actual commercial farm decision-making without field calibration with regional agricultural extension data.
> 2. **Single-User Academic Demo:** The application does not implement authentication (JWT, OAuth) or multi-tenant user access control by design, prioritizing simplicity and ease of review.
> 3. **Simplified Weather Model:** The precipitation forecast uses empirical rolling moving averages and linear regression over synthetic monthly index sequences; it does not connect to live Doppler radar or numerical weather prediction (NWP) satellite feeds.
