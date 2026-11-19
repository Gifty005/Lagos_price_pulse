# Lagos_price_pulse
Multi-factor machine learning system and Streamlit app for predicting monthly tomato and onion prices at  Lagos Markets.

# Lagos Price Pulse: Perishable Food Price Prediction (Mile 12 Market)

A multi-factor machine learning pipeline and interactive Streamlit web application that forecasts monthly retail/wholesale prices (₦/kg) for **tomatoes** and **onions** at Mile 12 Market, Lagos (2020–2025).



## Project Objectives
1. Integrate monthly food price records (WFP/NBS) with diesel/petrol prices (NBS), Northern Nigeria rainfall across Kano, Kaduna, and Jos (Open-Meteo), and NGN/USD exchange rates (CBN).
2. Analyze price shocks before and after major macroeconomic events (such as the June 2023 fuel subsidy removal) and seasonal harvest cycles.

3. Train and compare **Persistence Baseline**, **Linear Regression**, **Random Forest**, and **XGBoost** models using chronological train/test splits (MAE & RMSE).

4. Deploy an interactive Streamlit dashboard with a buyer/trader risk-alert engine (`GREEN`, `YELLOW`, `RED`).

## Repository Structure
text
Lagos_price_pulse/
├── data/
│   ├── raw/                  # Raw source CSVs (WFP/NBS food, fuel, rainfall, CBN FX)
│   └── processed/            # Cleaned monthly tables & merged master_mile12_data.csv
├── notebooks/                # Data cleaning & Exploratory Data Analysis (EDA) notebooks
├── src/                      # Feature engineering, model training, and risk engine modules
├── models/                   # Saved best-performing .joblib models
├── tests/                    # Pytest unit tests for pipeline and risk thresholds
├── app.py                    # Streamlit web application
└── requirements.txt          # Python dependencies
```

Risk Classification (src/risk_engine.py):

🟢 GREEN (< 5% projected increase): Low Risk / Stable Market

🟡 YELLOW (5% – 15% projected increase): Moderate Risk / Upward Price Pressure

🔴 RED (> 15% projected increase): High Risk / Severe Price Spike Alert
## Local Setup Instructions (For Team Collaborators)

1. **Clone the repository:**
   bash
   git clone [https://github.com/Gifty005/Lagos_price_pulse.git](https://github.com/Gifty005/Lagos_price_pulse.git)
   cd Lagos_price_pulse
   

2. **Create and activate a virtual environment:**
   * **Windows (PowerShell):**
     powershell
     python -m venv .venv
     .venv\Scripts\activate
    
   * **Mac / Linux:**
     bash
     python3 -m venv .venv
     source .venv/bin/activate
     

3. **Install dependencies:**
   bash
   pip install -r requirements.txt
   

4. **Run the Streamlit dashboard (once `app.py` is built):**
   bash
   streamlit run app.py
   
Team Credits
Gifty — Project Lead, Data Integration (master_mile12_data.csv), Model Benchmarking & Deployment

Damola — Exploratory Data Analysis (notebooks/eda.ipynb) & Feature Engineering (src/features.py)

Bisola — Chronological Model Training & Evaluation (src/train.py)

Yasir — Risk-Alert Engine (src/risk_engine.py) & Streamlit Dashboard (app.py)
