/**
 * Smart Agricultural Prediction System - Dashboard Controller
 * Fetches analytics data and renders interactive Chart.js visualizations
 */

document.addEventListener('DOMContentLoaded', () => {
    loadDashboardAnalytics();
    loadWeatherTrend();
});

let timelineChartInstance = null;
let yieldChartInstance = null;
let weatherChartInstance = null;

async function loadDashboardAnalytics() {
    try {
        const response = await fetch('/api/v1/dashboard-stats');
        if (!response.ok) throw new Error('Failed to load dashboard statistics');
        const data = await response.json();

        // Update Stat Cards if element exists
        const totalElem = document.getElementById('stat-total-predictions');
        if (totalElem) totalElem.textContent = data.total_predictions;

        const cropElem = document.getElementById('stat-common-crop');
        if (cropElem) cropElem.textContent = data.most_common_crop;

        const yieldElem = document.getElementById('stat-avg-yield');
        if (yieldElem) yieldElem.textContent = data.avg_predicted_yield ? `${data.avg_predicted_yield} kg/ha` : 'N/A';

        const alertsElem = document.getElementById('stat-disease-alerts');
        if (alertsElem) alertsElem.textContent = data.high_risk_disease_count;

        // Render Chart 1: Predictions Timeline
        renderTimelineChart(data.timeline);

        // Render Chart 2: Yield by Crop Type
        renderYieldByCropChart(data.yield_by_crop);

    } catch (err) {
        console.error('Error loading dashboard stats:', err);
    }
}

function renderTimelineChart(timelineData) {
    const ctx = document.getElementById('predictionsTimelineChart');
    if (!ctx) return;

    const labels = timelineData && Object.keys(timelineData).length > 0
        ? Object.keys(timelineData)
        : ['Day 1', 'Day 2', 'Day 3', 'Today'];
    const counts = timelineData && Object.keys(timelineData).length > 0
        ? Object.values(timelineData)
        : [0, 0, 0, 1];

    if (timelineChartInstance) {
        timelineChartInstance.destroy();
    }

    timelineChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Prediction Queries',
                data: counts,
                fill: true,
                backgroundColor: 'rgba(22, 163, 74, 0.12)',
                borderColor: '#16a34a',
                borderWidth: 2.5,
                pointBackgroundColor: '#15803d',
                pointRadius: 4,
                tension: 0.35
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { padding: 10, cornerRadius: 8 }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: { precision: 0 },
                    grid: { color: 'rgba(226, 232, 240, 0.6)' }
                },
                x: {
                    grid: { display: false }
                }
            }
        }
    });
}

function renderYieldByCropChart(cropYieldData) {
    const ctx = document.getElementById('yieldByCropChart');
    if (!ctx) return;

    // Fallback baseline reference values if no user yield predictions logged yet
    const defaultData = {
        'Wheat': 3850,
        'Rice': 4620,
        'Maize': 5480,
        'Cotton': 2390,
        'Sugarcane': 71500
    };

    const hasRealData = cropYieldData && Object.keys(cropYieldData).length > 0;
    const labels = hasRealData ? Object.keys(cropYieldData) : Object.keys(defaultData);
    const values = hasRealData ? Object.values(cropYieldData) : Object.values(defaultData);

    if (yieldChartInstance) {
        yieldChartInstance.destroy();
    }

    yieldChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: hasRealData ? 'Avg Predicted Yield (kg/ha)' : 'Baseline Yield Profile (kg/ha)',
                data: values,
                backgroundColor: [
                    '#16a34a',
                    '#0284c7',
                    '#f59e0b',
                    '#8b5cf6',
                    '#0d9488'
                ],
                borderRadius: 8,
                borderSkipped: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: true, position: 'top' },
                tooltip: {
                    callbacks: {
                        label: (item) => ` ${item.raw.toLocaleString()} kg/hectare`
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(226, 232, 240, 0.6)' },
                    ticks: {
                        callback: (val) => val >= 1000 ? `${(val / 1000).toFixed(0)}k` : val
                    }
                },
                x: { grid: { display: false } }
            }
        }
    });
}

async function loadWeatherTrend() {
    const ctx = document.getElementById('weatherForecastChart');
    if (!ctx) return;

    try {
        const resp = await fetch('/api/v1/weather-trend');
        if (!resp.ok) return;
        const res = await resp.json();
        const info = res.data;

        const labels = info.historical_series.map(s => s.month_name);
        const actuals = info.historical_series.map(s => s.rainfall_mm);
        const movingAvgs = info.historical_series.map(s => s.moving_avg_mm);

        // Append projected next month
        labels.push('Next (Proj)');
        actuals.push(null);
        movingAvgs.push(info.forecast_next_month_mm);

        const trendBadge = document.getElementById('weather-trend-badge');
        if (trendBadge) {
            trendBadge.textContent = `${info.trend_direction} (${info.forecast_next_month_mm} mm forecast)`;
            trendBadge.className = `badge ${info.trend_direction === 'Increasing' ? 'bg-success' : 'bg-info'}`;
        }

        if (weatherChartInstance) {
            weatherChartInstance.destroy();
        }

        weatherChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Historical Rainfall (mm)',
                        data: actuals,
                        borderColor: '#0284c7',
                        backgroundColor: 'rgba(2, 132, 199, 0.08)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3
                    },
                    {
                        label: '3-Month Moving Average / Forecast',
                        data: movingAvgs,
                        borderColor: '#f59e0b',
                        borderDash: [5, 5],
                        borderWidth: 2.5,
                        pointRadius: 4,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    tooltip: { cornerRadius: 8 }
                },
                scales: {
                    y: { beginAtZero: true, grid: { color: 'rgba(226, 232, 240, 0.6)' } },
                    x: { grid: { display: false } }
                }
            }
        });
    } catch (e) {
        console.error('Weather trend error:', e);
    }
}
