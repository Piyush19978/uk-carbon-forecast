# UK carbon-intensity forecaster + low-carbon charging window

Forecasts the carbon intensity of UK electricity (gCO2/kWh, half-hourly, up to 24 h ahead) with a **calibrated 80% prediction interval**, and uses the forecast to suggest the cleanest time to charge an EV.

****Notebook** (full analysis): https://colab.research.google.com/drive/1SFXbvsRY8RM-dKQJn8XIDYodUexm2suP#scrollTo=8QGFRH3NR3Cp

> Not an official forecast and not a safety tool. It uses *average* grid intensity, not *marginal* intensity (what your extra load actually causes).

## What it does
- Pulls the latest data from the Carbon Intensity API, cleans it with the same rules as training, and builds the same features the model was trained on.
- Quantile gradient boosting (10th / 50th / 90th percentile) predicts the *change* from the latest value; **conformal prediction (CQR)** calibrates the interval on a held-out year.
- Picks the cheapest contiguous charging block before a deadline.

## Data
Half-hourly carbon intensity and generation mix, Jan 2023 to Oct 2026, from the Carbon Intensity API (carbonintensity.org.uk; please check and respect its terms and attribution). Hourly weather from Open-Meteo was prepared but is **not** used in the deployed model.

### Data issues found (and handled)
- 122 missing timestamps (two long gaps of ~45 h and ~15 h) and 5 null actuals; 6 exact-zero readings that sit between normal values, treated as sensor glitches.
- Generation mix has a 9-12 day gap at the start of every year.
- The API's own `forecast` column behaves like a ~1 h nowcast (correlation 0.985 with the actual one hour earlier, only 0.63 at 24 h). Using it as a feature would have leaked information, so it is excluded.
- Gaps are filled only when the **whole** gap is <= 3 h, and only for features, never for labels. (A first version used `interpolate(limit=...)`, which silently fills the start of long gaps; caught during review of the output and fixed.)

## Method
- Time-based split with 24 h purge gaps: train Jan 2023 to Jun 2024, calibration Jul 2024 to Jun 2025 (a full year), test Jul 2025 to Oct 2026 (545k forecasts, never used for fitting or calibration).
- Features known at the forecast origin only (lags, rolling stats, generation mix, same time yesterday / last week, UK local calendar). Leakage checks are asserted in the notebook; a train/serve skew test confirms live features match training features (max difference 6e-12).
- Baselines: persistence, same time yesterday, same time last week, month x time-of-day climatology.

## Results (test period)

MAE in gCO2/kWh, best baseline vs model:

| Horizon | Best baseline | Model |
|---|---|---|
| 0.5 h | 4.64 | 8.86 |
| 1 h | 8.98 | 10.54 |
| 2 h | 16.39 | 14.44 |
| 4 h | 28.53 | 21.96 |
| 6 h | 36.73 | 27.45 |
| 12 h | 38.99 | 34.81 |
| 18 h | 38.98 | 39.51 |
| 24 h | 38.98 | 37.26 |

The model helps most at 2-12 h (11-25% lower error). In the first hour, repeating the latest value is better, and at 18 h the baseline is marginally better.

**Interval quality (nominal 80%):** raw quantiles covered 75.7%; CQR with per-horizon calibration covered **80.0%** (mean width about 101 gCO2/kWh). By month, coverage ranged from about 75% (June) to 83% (October): the conformal guarantee is marginal, not per-season.

**Charging scenario** (evening plug-in around 19:00, ready by about 08:00, 473 evenings):

| Charge time | Immediate | Naive forecast | Model forecast | Oracle |
|---|---|---|---|---|
| 2 h | 151.2 | 118.3 | 113.8 | 105.4 |
| 4 h | 143.4 | 118.9 | 114.1 | 108.8 |
| 6 h | 135.3 | 119.0 | 115.2 | 111.3 |

(mean gCO2/kWh of the energy drawn). The model captures 82-85% of the oracle's saving. Charging immediately lands on the evening peak, so the fairer comparison is the naive "same time yesterday" scheduler: the model is better by 4.5 / 4.8 / 3.8 gCO2/kWh (2 / 4 / 6 h), with 95% weekly-block-bootstrap intervals [2.3, 6.5], [3.3, 6.4], [2.7, 4.9].

## Limitations (read these)
- Average, not marginal, carbon intensity.
- 24 h accuracy is limited because no weather forecast is used.
- Hyperparameters were chosen on a single-season validation window (Apr-Jun 2024); they did not transfer perfectly (final test MAE 31.75 vs 30.85 for an earlier, less regularised model that was not selected afterwards, to avoid tuning on the test set).
- Direct-vs-delta target choice was first compared on the test set; the final configuration was then chosen on validation data only.
- Coverage varies by season (see above); no adaptive/online recalibration yet.

## Run locally
```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Files: `app.py` (UI), `predict_lib.py` (fetch, clean, features, forecast), `deploy/` (saved models and settings, trained with scikit-learn 1.6.1).
