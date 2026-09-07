"""
Weather Pattern Analysis Module
Provides moving-average trend analysis and naive next-month rainfall forecasting
using linear regression on month index.
"""

from typing import List, Dict, Any
import numpy as np
from sklearn.linear_model import LinearRegression

def generate_synthetic_monthly_rainfall(
    num_months: int = 12,
    base_rainfall: float = 110.0,
    seed: int = 101
) -> List[Dict[str, Any]]:
    """
    Generates realistic seasonal monthly rainfall history for a farm region.
    """
    np.random.seed(seed)
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]
    
    records = []
    for i in range(num_months):
        month_idx = i % 12
        # Monsoonal / seasonal bell-curve peak in summer/monsoon months (Jun-Sep)
        seasonal_factor = np.sin((month_idx - 2) * (np.pi / 6.0))
        seasonal_rainfall = max(15.0, base_rainfall + seasonal_factor * 65.0 + np.random.normal(0, 12.0))
        records.append({
            "month_index": i + 1,
            "month_name": month_names[month_idx],
            "rainfall_mm": round(float(seasonal_rainfall), 1)
        })
    return records

def analyze_rainfall_series(
    rainfall_records: List[Dict[str, Any]],
    window: int = 3
) -> Dict[str, Any]:
    """
    Computes rolling moving average, linear regression trend slope,
    and naive next-month projection.
    
    Args:
        rainfall_records: List of dicts containing 'rainfall_mm' and 'month_name'
        window: Moving average window size (default 3 months)
        
    Returns:
        Dict containing moving average series, slope, trend direction, and forecast.
    """
    if not rainfall_records:
        return {
            "error": "No rainfall records provided",
            "forecast_next_month_mm": 0.0,
            "trend_direction": "unknown"
        }
        
    values = np.array([r["rainfall_mm"] for r in rainfall_records], dtype=float)
    n = len(values)
    
    # 1. Moving Average
    moving_averages = []
    for i in range(n):
        if i + 1 < window:
            moving_averages.append(round(float(np.mean(values[:i + 1])), 1))
        else:
            moving_averages.append(round(float(np.mean(values[i + 1 - window:i + 1])), 1))
            
    # 2. Linear Regression Trend on Month Index
    X = np.arange(1, n + 1).reshape(-1, 1)
    y = values.reshape(-1, 1)
    
    reg = LinearRegression()
    reg.fit(X, y)
    
    slope = float(reg.coef_[0][0])
    intercept = float(reg.intercept_[0])
    
    # Naive Next-Month Forecast (Month n+1)
    next_month_idx = n + 1
    forecast_raw = float(reg.predict(np.array([[next_month_idx]]))[0][0])
    forecast_mm = max(5.0, round(forecast_raw, 1))
    
    if slope > 1.5:
        trend_direction = "Increasing"
    elif slope < -1.5:
        trend_direction = "Decreasing"
    else:
        trend_direction = "Stable"
        
    # Append moving average to returned series
    series_with_ma = []
    for r, ma in zip(rainfall_records, moving_averages):
        series_with_ma.append({
            "month_index": r.get("month_index"),
            "month_name": r.get("month_name"),
            "rainfall_mm": r.get("rainfall_mm"),
            "moving_avg_mm": ma
        })
        
    return {
        "historical_series": series_with_ma,
        "window_size": window,
        "trend_slope": round(slope, 3),
        "trend_direction": trend_direction,
        "forecast_next_month_mm": forecast_mm,
        "recent_avg_mm": round(float(np.mean(values[-min(3, n):])), 1)
    }

if __name__ == "__main__":
    synthetic_series = generate_synthetic_monthly_rainfall(12)
    analysis = analyze_rainfall_series(synthetic_series)
    print("Weather Trend Analysis Result:")
    print(f"Trend Direction: {analysis['trend_direction']} (Slope: {analysis['trend_slope']})")
    print(f"Next Month Forecast: {analysis['forecast_next_month_mm']} mm")
    for item in analysis["historical_series"]:
        print(f"  {item['month_name']}: {item['rainfall_mm']} mm (MA: {item['moving_avg_mm']})")
