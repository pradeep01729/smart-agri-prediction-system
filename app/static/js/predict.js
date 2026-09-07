/**
 * Smart Agricultural Prediction System - Prediction Controller
 * Handles client-side validation, AJAX fetch submission, inline animated results,
 * and quick-fill demo presets.
 */

document.addEventListener('DOMContentLoaded', () => {
    initYieldForm();
    initDiseaseForm();
    initFertilizerForm();
});

/* ==========================================================================
   1. CROP YIELD PREDICTION FORM
   ========================================================================== */
function initYieldForm() {
    const form = document.getElementById('yieldPredictionForm');
    if (!form) return;

    // Quick fill presets
    const presets = {
        wheat: { crop_type: 'wheat', soil_ph: 6.5, rainfall_mm: 580, temperature_c: 21, humidity_pct: 62, nitrogen: 120, phosphorus: 45, potassium: 40 },
        rice: { crop_type: 'rice', soil_ph: 6.8, rainfall_mm: 1300, temperature_c: 29, humidity_pct: 85, nitrogen: 140, phosphorus: 55, potassium: 60 },
        sugarcane: { crop_type: 'sugarcane', soil_ph: 7.0, rainfall_mm: 1500, temperature_c: 32, humidity_pct: 75, nitrogen: 180, phosphorus: 65, potassium: 85 }
    };

    document.querySelectorAll('[data-preset-yield]').forEach(btn => {
        btn.addEventListener('click', () => {
            const key = btn.getAttribute('data-preset-yield');
            const data = presets[key];
            if (!data) return;
            for (const [k, v] of Object.entries(data)) {
                const el = form.elements[k];
                if (el) el.value = v;
            }
        });
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const submitBtn = form.querySelector('button[type="submit"]');
        const alertBox = document.getElementById('yield-error-alert');
        const resultContainer = document.getElementById('yield-result-container');

        if (alertBox) alertBox.classList.add('d-none');

        // Extract payload
        const formData = new FormData(form);
        const payload = Object.fromEntries(formData.entries());

        // Basic client validation
        if (!payload.crop_type) {
            showError(alertBox, 'Please select a crop type.');
            return;
        }

        setButtonLoading(submitBtn, true);

        try {
            const response = await fetch('/api/v1/predict/yield', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await response.json();
            setButtonLoading(submitBtn, false);

            if (!response.ok || data.status !== 'success') {
                showError(alertBox, data.message || 'Error computing yield prediction');
                return;
            }

            // Render result inline
            renderYieldResult(data);
            if (resultContainer) {
                resultContainer.classList.remove('d-none');
                resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        } catch (err) {
            setButtonLoading(submitBtn, false);
            showError(alertBox, 'Network or server error occurred. Please try again.');
        }
    });
}

function renderYieldResult(data) {
    const valElem = document.getElementById('yield-result-val');
    const confElem = document.getElementById('yield-conf-val');
    const barElem = document.getElementById('yield-conf-bar');
    const intervalElem = document.getElementById('yield-interval-val');
    const cropElem = document.getElementById('yield-crop-name');

    if (valElem) valElem.textContent = `${Number(data.predicted_yield_kg_per_hectare).toLocaleString()} kg/ha`;
    if (confElem) confElem.textContent = `${data.confidence_pct}%`;
    if (barElem) barElem.style.width = `${data.confidence_pct}%`;
    if (cropElem) cropElem.textContent = data.crop_type.toUpperCase();

    if (intervalElem && data.confidence_interval) {
        intervalElem.textContent = `95% Confidence Interval: ${Number(data.confidence_interval[0]).toLocaleString()} — ${Number(data.confidence_interval[1]).toLocaleString()} kg/ha`;
    }
}

/* ==========================================================================
   2. DISEASE RISK PREDICTION FORM
   ========================================================================== */
function initDiseaseForm() {
    const form = document.getElementById('diseasePredictionForm');
    if (!form) return;

    const presets = {
        high_risk: { crop_type: 'rice', temperature_c: 28, humidity_pct: 92, rainfall_mm: 320, leaf_wetness_hours: 14, prior_disease_history: 1 },
        low_risk: { crop_type: 'wheat', temperature_c: 18, humidity_pct: 45, rainfall_mm: 30, leaf_wetness_hours: 2, prior_disease_history: 0 },
        medium_risk: { crop_type: 'maize', temperature_c: 25, humidity_pct: 72, rainfall_mm: 110, leaf_wetness_hours: 6, prior_disease_history: 0 }
    };

    document.querySelectorAll('[data-preset-disease]').forEach(btn => {
        btn.addEventListener('click', () => {
            const key = btn.getAttribute('data-preset-disease');
            const data = presets[key];
            if (!data) return;
            for (const [k, v] of Object.entries(data)) {
                const el = form.elements[k];
                if (el) el.value = v;
            }
        });
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const submitBtn = form.querySelector('button[type="submit"]');
        const alertBox = document.getElementById('disease-error-alert');
        const resultContainer = document.getElementById('disease-result-container');

        if (alertBox) alertBox.classList.add('d-none');

        const formData = new FormData(form);
        const payload = Object.fromEntries(formData.entries());

        setButtonLoading(submitBtn, true);

        try {
            const response = await fetch('/api/v1/predict/disease', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await response.json();
            setButtonLoading(submitBtn, false);

            if (!response.ok || data.status !== 'success') {
                showError(alertBox, data.message || 'Error predicting disease risk');
                return;
            }

            renderDiseaseResult(data);
            if (resultContainer) {
                resultContainer.classList.remove('d-none');
                resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        } catch (err) {
            setButtonLoading(submitBtn, false);
            showError(alertBox, 'Network or server error occurred.');
        }
    });
}

function renderDiseaseResult(data) {
    const badge = document.getElementById('disease-badge');
    const advisory = document.getElementById('disease-advisory');
    const confVal = document.getElementById('disease-conf-val');
    const confBar = document.getElementById('disease-conf-bar');
    const probContainer = document.getElementById('disease-probs-container');

    const risk = data.risk_level.toLowerCase();
    if (badge) {
        badge.textContent = `${data.risk_level.toUpperCase()} RISK`;
        badge.className = `result-badge badge-${risk}`;
    }

    if (advisory) advisory.textContent = data.advisory;
    if (confVal) confVal.textContent = `${data.confidence_pct}%`;
    if (confBar) confBar.style.width = `${data.confidence_pct}%`;

    if (probContainer && data.class_probabilities) {
        probContainer.innerHTML = '';
        for (const [cls, pct] of Object.entries(data.class_probabilities)) {
            const col = document.createElement('div');
            col.className = 'mb-2';
            col.innerHTML = `
                <div class="d-flex justify-content-between small mb-1">
                    <span class="text-capitalize fw-semibold">${cls} Risk</span>
                    <span>${pct}%</span>
                </div>
                <div class="confidence-bar-wrapper" style="height: 6px;">
                    <div class="confidence-bar-fill" style="width: ${pct}%;"></div>
                </div>
            `;
            probContainer.appendChild(col);
        }
    }
}

/* ==========================================================================
   3. FERTILIZER RECOMMENDATION FORM
   ========================================================================== */
function initFertilizerForm() {
    const form = document.getElementById('fertilizerPredictionForm');
    if (!form) return;

    const presets = {
        low_nitrogen: { crop_type: 'maize', soil_ph: 6.5, nitrogen: 25, phosphorus: 45, potassium: 50, moisture_pct: 55 },
        low_phosphorus: { crop_type: 'wheat', soil_ph: 6.8, nitrogen: 90, phosphorus: 15, potassium: 55, moisture_pct: 48 },
        acidic_soil: { crop_type: 'rice', soil_ph: 4.9, nitrogen: 75, phosphorus: 35, potassium: 40, moisture_pct: 60 }
    };

    document.querySelectorAll('[data-preset-fertilizer]').forEach(btn => {
        btn.addEventListener('click', () => {
            const key = btn.getAttribute('data-preset-fertilizer');
            const data = presets[key];
            if (!data) return;
            for (const [k, v] of Object.entries(data)) {
                const el = form.elements[k];
                if (el) el.value = v;
            }
        });
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const submitBtn = form.querySelector('button[type="submit"]');
        const alertBox = document.getElementById('fertilizer-error-alert');
        const resultContainer = document.getElementById('fertilizer-result-container');

        if (alertBox) alertBox.classList.add('d-none');

        const formData = new FormData(form);
        const payload = Object.fromEntries(formData.entries());

        setButtonLoading(submitBtn, true);

        try {
            const response = await fetch('/api/v1/predict/fertilizer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await response.json();
            setButtonLoading(submitBtn, false);

            if (!response.ok || data.status !== 'success') {
                showError(alertBox, data.message || 'Error generating recommendation');
                return;
            }

            renderFertilizerResult(data);
            if (resultContainer) {
                resultContainer.classList.remove('d-none');
                resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        } catch (err) {
            setButtonLoading(submitBtn, false);
            showError(alertBox, 'Network or server error occurred.');
        }
    });
}

function renderFertilizerResult(data) {
    const badge = document.getElementById('fertilizer-badge');
    const info = document.getElementById('fertilizer-info');
    const confVal = document.getElementById('fertilizer-conf-val');
    const confBar = document.getElementById('fertilizer-conf-bar');
    const probContainer = document.getElementById('fertilizer-probs-container');

    if (badge) badge.textContent = `Recommended: ${data.recommended_fertilizer}`;
    if (info) info.textContent = data.fertilizer_info;
    if (confVal) confVal.textContent = `${data.confidence_pct}%`;
    if (confBar) confBar.style.width = `${data.confidence_pct}%`;

    if (probContainer && data.class_probabilities) {
        probContainer.innerHTML = '';
        for (const [cls, pct] of Object.entries(data.class_probabilities)) {
            const col = document.createElement('div');
            col.className = 'mb-2';
            col.innerHTML = `
                <div class="d-flex justify-content-between small mb-1">
                    <span class="fw-semibold">${cls}</span>
                    <span>${pct}%</span>
                </div>
                <div class="confidence-bar-wrapper" style="height: 6px;">
                    <div class="confidence-bar-fill" style="width: ${pct}%;"></div>
                </div>
            `;
            probContainer.appendChild(col);
        }
    }
}

/* ==========================================================================
   UTILITY HELPERS
   ========================================================================== */
function showError(alertBox, message) {
    if (!alertBox) return;
    alertBox.textContent = message;
    alertBox.classList.remove('d-none');
}

function setButtonLoading(btn, isLoading) {
    if (!btn) return;
    if (isLoading) {
        btn.disabled = true;
        btn.dataset.originalHtml = btn.innerHTML;
        btn.innerHTML = `<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Computing ML Model...`;
    } else {
        btn.disabled = false;
        if (btn.dataset.originalHtml) {
            btn.innerHTML = btn.dataset.originalHtml;
        }
    }
}
