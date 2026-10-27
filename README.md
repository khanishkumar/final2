# AI-Based Heat Exchanger Fouling Prediction, Performance Monitoring & Maintenance Support Dashboard

Streamlit dashboard for an **automotive oil-cooler-based, PHE-like heat-exchanger prototype**
(hot fluid: water, cold fluid: coolant; fouling of interest: CaCO₃ hard-water scaling).
It is *not* a commercial plate heat exchanger.

Sections: 1 Data Input · 2 Data Preprocessing · 3 Engineering Performance (Rf) ·
4 Fouling Prediction & SHAP · 5 Maintenance Support.

## Repository structure
```
fouling-dashboard/
├── app.py              # Streamlit UI (entry point)
├── requirements.txt
├── README.md
├── .streamlit/config.toml   (optional)
└── hxlib/
    ├── __init__.py
    ├── data.py         # loading, validation, column detection
    ├── engineering.py  # Rf = 1/Uf - 1/Uclean, threshold, severity
    ├── models.py       # leakage-safe pipelines, LR / RF / XGBoost, metrics
    ├── explain.py      # SHAP
    └── maintenance.py  # rule-based decision support
```
Do **not** commit the dataset; upload it inside the running app.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
## Google Colab (development only)
```python
!pip install -q -r requirements.txt
!streamlit run app.py &>/content/log.txt &
```
then expose it with a tunnel (e.g. `npx localtunnel --port 8501`). This is only for testing; the final deployment uses Streamlit Community Cloud.

## Deploy on Streamlit Community Cloud
1. Create a GitHub account → **New repository** (e.g. `fouling-dashboard`, public, or private if your Streamlit workspace allows).
2. Upload `app.py`, `requirements.txt`, `README.md` and the `hxlib/` folder (GitHub web: *Add file → Upload files*; keep the folder structure), or push with git:
   `git init && git add . && git commit -m "init" && git branch -M main && git remote add origin <url> && git push -u origin main`
3. Go to https://share.streamlit.io → sign in with GitHub → **Create app** → *Deploy a public app from GitHub*.
4. Choose the repository, branch `main`, **Main file path: `app.py`**.
5. Open **Advanced settings** → choose **Python 3.11** (best wheel support for shap/xgboost) → Deploy.
6. When the app is live, open Section 1 and upload `heat_exchanger_fouling_data.csv`.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: hxlib` | `hxlib/` (with `__init__.py`) must be at the repo root next to `app.py`. |
| `ModuleNotFoundError: xgboost/shap` | Check spelling in `requirements.txt`, reboot the app (Manage app → Reboot). XGBoost is optional: the app still runs without it. |
| Build fails compiling shap/numba | Select Python 3.11 in Advanced settings; avoid Python 3.14. |
| App out of memory / slow | Lower RF trees, use fewer rows for global SHAP, or sample the dataset. |
| Upload rejected | Streamlit's default limit is 200 MB; the CSV must be UTF-8/Latin-1 text. |
| Stale behaviour after code push | Manage app → Reboot / Clear cache. |

## Engineering notes
* Rf = 1/Uf − 1/Uclean (m²·K/W); both U values are converted to W/(m²·K) and must share one area basis.
* Threshold 0.043 m²·K/kW (= 4.3×10⁻⁵ m²·K/W) is **provisional and unverified**; it is configurable.
* The threshold comparison is an engineering indicator, not proof of cleanliness or fouling.
* SHAP explains model behaviour, not physical causation.
* Forecasting (LSTM) is not implemented: it needs a timestamped series.
