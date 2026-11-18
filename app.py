from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.risk_engine import evaluate_price_risk

# ---------------------------------------------------------------------------
# Page Config & Custom Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Lagos Price Pulse | Mile 12 Market Risk Dashboard",
    page_icon="🌶️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .risk-banner {
        padding: 1.1rem 1.4rem;
        border-radius: 0.6rem;
        margin: 0.75rem 0 1.25rem 0;
        border-left: 8px solid;
    }
    .risk-GREEN {
        background-color: rgba(22, 163, 74, 0.14);
        border-color: #16a34a;
    }
    .risk-YELLOW {
        background-color: rgba(234, 179, 8, 0.16);
        border-color: #eab308;
    }
    .risk-RED {
        background-color: rgba(220, 38, 38, 0.16);
        border-color: #dc2626;
    }
    .risk-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .risk-body {
        font-size: 0.98rem;
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data & Model Loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_data() -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    data_path = Path("data/processed/master_mile12_data.csv")
    metrics_path = Path("models/model_metrics.csv")

    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    metrics_df = pd.read_csv(metrics_path) if metrics_path.exists() else None
    return df, metrics_df


@st.cache_resource
def load_models() -> dict[str, object]:
    return {
        "Tomatoes": joblib.load("models/tomato_model.joblib"),
        "Onions": joblib.load("models/onion_model.joblib"),
    }


def build_next_month_features(
    df: pd.DataFrame,
    sim_diesel: float,
    sim_fx: float,
    sim_rain: float,
) -> pd.DataFrame:
    """
    Constructs the exact feature vector defined in src/features.py for
    forecasting the next month (t + 1) using simulated shock values.
    """
    work_df = df.copy()

    # Apply simulated Diesel & USD/NGN shocks to the latest observed month (t)
    # so shift(1) passes them into next month's (t + 1) lag features.
    work_df.loc[work_df.index[-1], "diesel_price_ngn"] = sim_diesel
    work_df.loc[work_df.index[-1], "usd_ngn_rate"] = sim_fx

    # Append a placeholder row for next month (t + 1) so shift(1) and shift(2)
    # align to Dec 2025 (t) and Nov 2025 (t - 1).
    next_date = pd.to_datetime(work_df["date"].iloc[-1]) + pd.DateOffset(months=1)
    future_row = work_df.iloc[[-1]].copy()
    future_row["date"] = next_date
    work_df = pd.concat([work_df, future_row], ignore_index=True)

    # Exact feature engineering logic from src/features.py
    work_df["tomato_price_lag1"] = work_df["tomato_price_ngn"].shift(1)
    work_df["tomato_price_lag2"] = work_df["tomato_price_ngn"].shift(2)
    work_df["onion_price_lag1"] = work_df["onion_price_ngn"].shift(1)
    work_df["onion_price_lag2"] = work_df["onion_price_ngn"].shift(2)

    work_df["diesel_price_lag1"] = work_df["diesel_price_ngn"].shift(1)
    work_df["petrol_price_lag1"] = work_df["petrol_price_ngn"].shift(1)
    work_df["usd_ngn_lag1"] = work_df["usd_ngn_rate"].shift(1)
    work_df["north_rainfall_lag2"] = work_df["north_rainfall_mm"].shift(2)

    # Apply simulated Northern Rainfall shock directly to the active rainfall lag
    # so the rainfall slider updates next month's prediction immediately.
    work_df.loc[work_df.index[-1], "north_rainfall_lag2"] = sim_rain

    work_df["month"] = pd.to_datetime(work_df["date"]).dt.month
    work_df["is_rainy_season"] = work_df["month"].apply(
        lambda m: 1 if 5 <= m <= 9 else 0
    )

    return work_df.iloc[[-1]]


def predict_next_month(
    next_row: pd.DataFrame,
    model: object,
    target_col: str,
) -> float:
    """Aligns next-month features with the saved model and predicts price."""
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
        X_next = next_row[expected_cols].copy()
    else:
        default_features = [
            "tomato_price_lag1",
            "tomato_price_lag2",
            "onion_price_lag1",
            "onion_price_lag2",
            "diesel_price_lag1",
            "petrol_price_lag1",
            "usd_ngn_lag1",
            "north_rainfall_lag2",
            "month",
            "is_rainy_season",
        ]
        available = [c for c in default_features if c in next_row.columns]
        X_next = next_row[available].copy()

    pred = float(model.predict(X_next)[0])
    return max(pred, 0.0)


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
def main() -> None:
    df, metrics_df = load_data()
    models = load_models()

    latest_row = df.iloc[-1]
    latest_date = pd.to_datetime(latest_row["date"])
    next_month_date = latest_date + pd.DateOffset(months=1)

    # -----------------------------------------------------------------------
    # Sidebar: Commodity & "What-If" Shock Simulator
    # -----------------------------------------------------------------------
    st.sidebar.header("🎛️ Simulation Controls")
    commodity = st.sidebar.selectbox(
        "Select Mile 12 Commodity",
        options=["Tomatoes", "Onions"],
        index=0,
    )

    target_col = (
        "tomato_price_ngn" if commodity == "Tomatoes" else "onion_price_ngn"
    )
    model = models[commodity]

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ What-If Shock Simulator")
    st.sidebar.caption(
        f"Adjust macro & climate drivers from **{latest_date.strftime('%b %Y')}** "
        "baseline values to test next-month price sensitivity."
    )

    base_diesel = float(latest_row["diesel_price_ngn"])
    base_fx = float(latest_row["usd_ngn_rate"])
    base_rain = float(df["north_rainfall_mm"].iloc[-2])

    sim_diesel = st.sidebar.slider(
        "Diesel Price (₦ / Litre)",
        min_value=max(100.0, round(base_diesel * 0.5, -1)),
        max_value=round(max(base_diesel * 1.8, 2500.0), -1),
        value=round(base_diesel, 1),
        step=10.0,
    )

    sim_fx = st.sidebar.slider(
        "USD/NGN Exchange Rate (₦)",
        min_value=max(300.0, round(base_fx * 0.5, -1)),
        max_value=round(max(base_fx * 1.8, 2500.0), -1),
        value=round(base_fx, 1),
        step=10.0,
    )

    max_rain_bound = max(float(df["north_rainfall_mm"].max()) * 1.5, 350.0)
    sim_rain = st.sidebar.slider(
        "Northern Rainfall (mm)",
        min_value=0.0,
        max_value=round(max_rain_bound, 0),
        value=round(base_rain, 1),
        step=5.0,
    )

    if st.sidebar.button("Reset to Baseline Values", use_container_width=True):
        st.rerun()

    next_features = build_next_month_features(df, sim_diesel, sim_fx, sim_rain)
    current_price = float(latest_row[target_col])
    predicted_price = predict_next_month(next_features, model, target_col)
    risk = evaluate_price_risk(current_price, predicted_price)

    # -----------------------------------------------------------------------
    # Header & KPI Cards
    # -----------------------------------------------------------------------
    st.title("🌶️ Lagos Price Pulse — Mile 12 Early-Warning System")
    st.caption(
        f"Forecasting wholesale food inflation for **{commodity}** using "
        "NBS fuel prices, FX rates, and Northern Nigeria rainfall data."
    )

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric(
        label=f"Current Price ({latest_date.strftime('%b %Y')})",
        value=f"₦{risk['current_price']:,.2f}",
    )
    kpi2.metric(
        label=f"Predicted Price ({next_month_date.strftime('%b %Y')})",
        value=f"₦{risk['predicted_price']:,.2f}",
        delta=f"{risk['pct_change']:+.2f}%",
        delta_color="inverse",
    )
    kpi3.metric(
        label="Risk Classification",
        value=f"{risk['status']} TIER",
        delta=f"Diesel: ₦{sim_diesel:,.0f} | FX: ₦{sim_fx:,.0f}",
        delta_color="off",
    )

    # -----------------------------------------------------------------------
    # Color-Coded Live Risk Banner
    # -----------------------------------------------------------------------
    status_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}[risk["status"]]
    st.markdown(
        f"""
        <div class="risk-banner risk-{risk['status']}">
            <div class="risk-title">{status_emoji} {risk['status']} ALERT — {risk['headline']} ({risk['pct_change']:+.2f}%)</div>
            <p class="risk-body"><strong>Actionable Advisory:</strong> {risk['recommendation']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # Historical Trend (2020–2025) + Subsidy Marker + Forecast Point
    # -----------------------------------------------------------------------
    st.subheader(f"📈 {commodity} Price Trajectory (2020–2025) & Next-Month Forecast")

    tier_colors = {"GREEN": "#16a34a", "YELLOW": "#eab308", "RED": "#dc2626"}
    forecast_color = tier_colors[risk["status"]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df[target_col],
            mode="lines+markers",
            name="Historical Mile 12 Price",
            line=dict(color="#2563eb", width=2.5),
            marker=dict(size=5),
            hovertemplate="<b>%{x|%b %Y}</b><br>Price: ₦%{y:,.2f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[latest_date, next_month_date],
            y=[current_price, predicted_price],
            mode="lines",
            name="Projected Trajectory",
            line=dict(color=forecast_color, width=2.5, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[next_month_date],
            y=[predicted_price],
            mode="markers+text",
            name=f"Forecast ({next_month_date.strftime('%b %Y')})",
            text=[f"₦{predicted_price:,.0f} ({risk['pct_change']:+.1f}%)"],
            textposition="top center",
            marker=dict(
                color=forecast_color,
                size=14,
                symbol="diamond",
                line=dict(color="white", width=1.5),
            ),
            hovertemplate=(
                f"<b>Forecast ({next_month_date.strftime('%b %Y')})</b><br>"
                "Predicted: ₦%{y:,.2f}<br>"
                f"Change: {risk['pct_change']:+.2f}%<extra></extra>"
            ),
        )
    )

    subsidy_date = pd.Timestamp("2023-06-01")
    if df["date"].min() <= subsidy_date <= next_month_date:
        fig.add_vline(
            x=subsidy_date.timestamp() * 1000,
            line_width=2,
            line_dash="dash",
            line_color="#f97316",
            annotation_text="June 2023: Fuel Subsidy Removal & FX Float",
            annotation_position="top left",
            annotation_font=dict(color="#f97316", size=12),
        )

    fig.update_layout(
        height=460,
        xaxis_title="Month",
        yaxis_title="Average Retail Price (₦)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------------------------
    # Model Benchmark Table
    # -----------------------------------------------------------------------
    st.subheader("🧪 2025 Out-of-Sample Model Benchmarks (MAE & RMSE)")
    if metrics_df is not None:
        display_metrics = metrics_df.copy()
        for col in display_metrics.columns:
            if col.lower() in {"commodity", "crop", "item", "target"}:
                filtered = display_metrics[
                    display_metrics[col]
                    .astype(str)
                    .str.lower()
                    .str.contains(commodity.lower()[:5])
                ]
                if not filtered.empty:
                    display_metrics = filtered
                break

        st.dataframe(display_metrics, use_container_width=True, hide_index=True)
    else:
        st.warning("`models/model_metrics.csv` not found in the repository.")


if __name__ == "__main__":
    main()