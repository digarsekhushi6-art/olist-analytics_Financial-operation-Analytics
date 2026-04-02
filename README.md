# Olist Financial Operations Analytics

End-to-end analytics project covering **revenue forecasting**, **churn analysis**, and **profitability** using the Brazilian E-Commerce (Olist) dataset.

**Stack:** Python · pandas · Prophet · XGBoost · SQLite · Streamlit · Plotly

Live Demo: https://olist-analyticsfinancial-operation-analytics-xahw4xrznx3ub7mcq.streamlit.app

---

## Project structure

```
olist-analytics/
├── data/
│   ├── raw/                  ← place Kaggle CSVs here (or auto-generated)
│   └── processed/            ← model outputs (auto-created)
├── src/
│   ├── generate_data.py      ← synthetic data generator (no Kaggle needed)
│   ├── etl.py                ← load, clean, merge, save to SQLite
│   └── models/
│       ├── forecast.py       ← Prophet revenue forecasting
│       ├── churn.py          ← XGBoost churn prediction
│       └── profitability.py  ← margin, seller, state, cohort analysis
├── app/
│   └── streamlit_app.py      ← 5-page interactive dashboard
├── .streamlit/config.toml
├── run_all.py                ← one-shot pipeline runner
├── requirements.txt
└── packages.txt              ← for Streamlit Cloud
```

---

## Quick start (local)

### Step 1 — Clone / download project

```bash
git clone https://github.com/YOUR_USERNAME/olist-analytics.git
cd olist-analytics
```

### Step 2 — Create virtual environment

```bash
python -m venv .venv

# Mac/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> Prophet requires additional system dependencies on some machines.
> If installation fails: `pip install prophet --no-build-isolation`

### Step 4A — Use real Kaggle data (recommended)

1. Go to: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
2. Download and unzip into `data/raw/`
3. You should have 9 CSV files named `olist_*_dataset.csv`

### Step 4B — Use synthetic data (instant, no Kaggle account needed)

```bash
python src/generate_data.py
```

This generates ~99K realistic orders with correct seasonality, distributions, and relationships.

### Step 5 — Run full pipeline

```bash
python run_all.py
```

This runs in sequence:
1. ETL → builds SQLite database with fact/dim tables
2. Forecast model → Prophet 6-month GMV forecast
3. Churn model → XGBoost customer risk scoring
4. Profitability → category margins, seller scorecards, cohort matrix

### Step 6 — Launch dashboard

```bash
streamlit run app/streamlit_app.py
```

Open: http://localhost:8501

---

## Deploy to Streamlit Cloud (free, public URL)

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/olist-analytics.git
git push -u origin main
```

**Important:** The `.gitignore` excludes raw CSVs and the database.
For deployment, the app auto-generates synthetic data on first run.

Add this to the top of `run_all.py` or handle in `streamlit_app.py` if deploying with auto-setup.

### Step 2 — Connect to Streamlit Cloud

1. Go to: https://share.streamlit.io
2. Sign in with GitHub
3. Click **"New app"**
4. Set:
   - **Repository:** `YOUR_USERNAME/olist-analytics`
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
5. Click **Deploy**

### Step 3 — Add secrets (optional)

For production with real data, add to Streamlit Secrets:
```toml
[data]
source = "synthetic"   # or "kaggle"
```

---

## Deploy to other platforms

### Render (free tier)

1. Create a `render.yaml` in project root:
```yaml
services:
  - type: web
    name: olist-analytics
    env: python
    buildCommand: pip install -r requirements.txt && python run_all.py
    startCommand: streamlit run app/streamlit_app.py --server.port $PORT
```
2. Push to GitHub, connect repo on render.com

### Railway

```bash
railway login
railway init
railway up
```
Set env var: `RAILWAY_DOCKERFILE_PATH=Dockerfile`

### Docker (self-hosted)

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN python run_all.py
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t olist-analytics .
docker run -p 8501:8501 olist-analytics
```

---

## Dashboard pages

| Page | What it shows |
|------|---------------|
| 🏠 Overview | GMV trend, order volume, customer segments |
| 📈 Revenue Forecasting | Prophet forecast vs actuals, MoM growth, 6-month projection |
| 🔄 Churn Analysis | RFM segments, churn probability, feature importance |
| 💰 Profitability | Margin by category, seller scorecards, freight by state |
| 📋 Cohort Retention | Monthly retention heatmap matrix |

---

## Key metrics produced

- **Revenue CAGR:** ~34% (2016–2018)
- **Forecast MAPE:** ~6–8% (Prophet with BR holidays)
- **Churn rate:** 96.9% (one-time buyers dominant)
- **Churn model AUC-ROC:** 0.84 (XGBoost)
- **Top margin category:** bed_bath_table ~68%
- **Highest freight burden:** Northern states (AM, AC, RO) 25–38% of order value

---

## Skills demonstrated

- Python ETL with pandas (multi-table merge, validation, derived columns)
- SQL data modeling (fact/dim/mart pattern, window functions, CTEs)
- Time-series forecasting (Prophet, seasonality decomposition)
- ML classification (XGBoost, feature importance, SHAP, AUC-ROC)
- RFM customer segmentation
- Cohort analysis & retention modeling
- Interactive dashboard deployment (Streamlit + Plotly)
