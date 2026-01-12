
import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from keras.models import load_model
import plotly.graph_objects as go
import joblib


@st.cache_resource(show_spinner=False)
def load_forecast_model(model_path):
    return load_model(model_path, compile=False)
# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Tesla Stock Forecast Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# LOAD CSS (Glassmorphism)
# ======================================================
with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ======================================================
# PATHS
# ======================================================
ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "TSLA.csv"
MODEL_DIR = ROOT / "models"
REPORT_PATH = ROOT / "reports" / "best_model_report.json"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
METRICS_CSV = ROOT / "reports" / "model_metrics_summary.csv"
metrics_df = pd.read_csv(METRICS_CSV)
metrics_df.set_index("Model", inplace=True)

st.markdown(
    """
    <div style="text-align:center; margin:30px 0;">
        <h1 class="section-title">Tesla Stock Forecast Dashboard</h1>
        <p style="color:#9ca3af; max-width:720px; margin:0 auto;">
            AI-driven 30-day price forecasting using deep learning models.
            Insights include price trends, forecast confidence, volatility, and risk metrics.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

def calculate_model_confidence(row, forecast_days, volatility):

    weighted_mape = (
        0.5 * row["MAPE_1D"] +
        0.3 * row["MAPE_5D"] +
        0.2 * row["MAPE_10D"]
    )

    accuracy_score = max(0, 100 - weighted_mape * 12)

    mape_std = np.std([
        row["MAPE_1D"],
        row["MAPE_5D"],
        row["MAPE_10D"]
    ])
    stability_score = max(0, 100 - mape_std * 15)

    avg_error = (
        row["MAE_1D"] +
        row["MAE_5D"] +
        row["MAE_10D"]
    ) / 3
    error_score = max(0, 100 - avg_error / 2)

    horizon_penalty = min(forecast_days * 1.5, 30)
    volatility_penalty = min(volatility * 120, 20)

    confidence = (
        0.35 * accuracy_score +
        0.25 * stability_score +
        0.20 * error_score +
        0.20 * (100 - horizon_penalty - volatility_penalty)
    )

    return round(max(40, min(confidence, 95)), 1)




# ======================================================
# LOAD DATA
# ======================================================
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df.sort_values("Date", inplace=True)

LOOKBACK_DAYS = 60
recent_df = df.tail(LOOKBACK_DAYS).copy()

close_prices = df["Close"].values.reshape(-1, 1)
last_close = float(close_prices[-1][0])

# ======================================================
# LOAD SCALER
# ======================================================
scaler = joblib.load(SCALER_PATH)
scaled_close = scaler.transform(close_prices)

# ======================================================
# LOAD BEST MODEL INFO
# ======================================================
with open(REPORT_PATH) as f:
    best_model_info = json.load(f)

best_model_name = best_model_info["model_name"]

# ======================================================
# MODEL REGISTRY
# ======================================================
model_files = {
    "LSTM_Tuned": "lstm_tuned.h5",
    "LSTM_Baseline": "lstm_baseline_best.h5",
    "SimpleRNN_Tuned": "simple_rnn_tuned.h5",
    "SimpleRNN_Baseline": "simple_rnn_baseline_best.h5",
}
recent_df["Daily Return %"] = recent_df["Close"].pct_change() * 100
recent_df["Volatility"] = recent_df["Daily Return %"].rolling(10).std()


# ======================================================
# SIDEBAR — ADVANCED FORECAST CONTROL PANEL
# ======================================================

import time

with st.sidebar:

    # --------------------------------------------------
    # SESSION STATE INIT (SAFE & REQUIRED)
    # --------------------------------------------------
    if "sidebar_loaded" not in st.session_state:
        st.session_state.sidebar_loaded = False

    if "prev_model" not in st.session_state:
        st.session_state.prev_model = best_model_name

    if "prev_days" not in st.session_state:
        st.session_state.prev_days = 10

    if "prev_compare" not in st.session_state:
        st.session_state.prev_compare = False
        
    if "is_rendering" not in st.session_state:
        st.session_state.is_rendering = False

    # --------------------------------------------------
    # INITIAL SIDEBAR LOADER (ONCE PER SESSION)
    # --------------------------------------------------
    if not st.session_state.sidebar_loaded:
        with st.spinner("Initializing forecast controls..."):
            time.sleep(2.5)
        st.session_state.sidebar_loaded = True

    # --------------------------------------------------
    # HEADER
    # --------------------------------------------------
    st.markdown(
        """
        <div class="sidebar-header">
            <div class="sidebar-main-title">📈 Forecast Control Panel</div>
            <div class="sidebar-description">
                Configure forecasting parameters and monitor model status.
            </div>
        </div>
        <div class="sidebar-divider"></div>
        """,
        unsafe_allow_html=True
    )
   # ==================================================
    # SECTION 1 — MODEL SELECTION MODE
    # ==================================================
    st.markdown(
        """
        <div class="sidebar-section-title">🧠 Model Selection</div>
        <div class="sidebar-section-sub">
            Choose automatic best model selection or manually select any trained model.
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Selection mode ---
    model_mode = st.radio(
        "Model Selection Mode",
        ["Use Best Model (Recommended)", "Manual Selection"],
        index=0,
        help="Best model is selected based on validation performance"
    )

    # --- Resolve active model ---
    if model_mode == "Use Best Model (Recommended)":
        active_model = best_model_name
        st.session_state.prev_model = active_model

        st.markdown(
            f"""
            <div style="margin-top:6px; font-size:13px; color:#9ca3af;">
                Automatically using best model:<br/>
                <b>{best_model_name}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
      

        selected_model = st.selectbox(
            "Select Model",
            list(model_files.keys()),
            index=list(model_files.keys()).index(st.session_state.prev_model),
            key="manual_model_select",
          
            disabled=st.session_state.is_rendering,
            help="Manually select any trained model"
        )

        if selected_model != st.session_state.prev_model:
            active_model = selected_model
            st.session_state.prev_model = active_model
        else:
            active_model = st.session_state.prev_model

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    model_path = MODEL_DIR / model_files[active_model]

    if (
        "loaded_model_name" not in st.session_state
        or st.session_state.loaded_model_name != active_model
    ):
        with st.spinner("Loading model & recalculating forecasts..."):
            st.session_state.model = load_forecast_model(model_path)
            st.session_state.loaded_model_name = active_model

        # ✅ SAFE PLACE TO UNLOCK UI
        st.session_state.is_rendering = False


    model = st.session_state.model
    # ==================================================
    # SECTION 2 — FORECAST HORIZON
    # ==================================================
    st.markdown(
        """
        <div class="sidebar-section-title">⏳ Forecast Horizon</div>
        <div class="sidebar-section-sub">
            Number of future trading days to predict.
        </div>
        """,
        unsafe_allow_html=True
    )

    forecast_days = st.slider(
        "Days Ahead",
        1, 30,
        st.session_state.prev_days,
        help="Longer horizons increase uncertainty"
    )

    if forecast_days != st.session_state.prev_days:
        with st.spinner("Updating forecast horizon..."):
            time.sleep(1.5)
        st.session_state.prev_days = forecast_days

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    # ==================================================
    # SECTION 3 — ANALYSIS OPTIONS
    # ==================================================
    st.markdown(
        """
        <div class="sidebar-section-title">📊 Analysis Options</div>
        <div class="sidebar-section-sub">
            Enable advanced analytical views.
        </div>
        """,
        unsafe_allow_html=True
    )

    show_model_comparison = st.toggle(
        "Enable Multi-Model Comparison",
        value=st.session_state.prev_compare,
        help="Overlay forecasts from all models"
    )

    if show_model_comparison != st.session_state.prev_compare:
        with st.spinner("Updating analysis options..."):
            time.sleep(1)
        st.session_state.prev_compare = show_model_comparison

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    # ==================================================
    # SECTION 4 — MODEL STATUS (TEXT STYLE, NO CARDS)
    # ==================================================
    

    model_confidence = {}
    avg_volatility = recent_df["Volatility"].dropna().mean()

    for model_name, row in metrics_df.iterrows():
        model_confidence[model_name] = calculate_model_confidence(
            row=row,
            forecast_days=forecast_days,
            volatility=avg_volatility
        )
    
    confidence_score = model_confidence.get(active_model, None)    


    st.markdown(
        """
        <div class="sidebar-section-title">📌 Model Status</div>
        <div class="sidebar-section-sub">
            Live system information based on current inputs.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="margin-top:6px; font-size:13px; color:#e5e7eb;">
            <b>Active Model:</b> {active_model}<br/>
            <b>Model Confidence:</b> {confidence_score:.1f}% if confidence_score is not None else "N/A"
            <span style="color:#9ca3af;">
                Confidence derived from MAPE, horizon length, and volatility
            </span>
            <span style="color:#9ca3af;">
                Confidence adjusted for recent volatility
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ==================================================
    # SIDEBAR FOOTER
    # ==================================================
    st.markdown(
        """
        <div class="sidebar-divider"></div>
        <div class="sidebar-footer">
            Forecasts update automatically<br/>
            when inputs change.
        </div>
        """,
        unsafe_allow_html=True
    )
    
   


# ======================================================
# FORECAST FUNCTION (SCALED → REAL)
# ======================================================
LOOKBACK = 60

def forecast_prices(series_scaled, days):
    X = series_scaled[-LOOKBACK:].reshape(1, LOOKBACK, 1)

    preds_scaled = model.predict(X, verbose=0).flatten()

    preds_scaled = preds_scaled[:days].reshape(-1, 1)
    return scaler.inverse_transform(preds_scaled).flatten()

# ======================================================
# RUN FORECAST
# ======================================================


future_dates = pd.date_range(
    start=recent_df["Date"].iloc[-1] + pd.Timedelta(days=1),
    periods=forecast_days
)

MIN_TABLE_DAYS = 10
table_days = max(MIN_TABLE_DAYS, forecast_days)

MAX_DAYS = max(forecast_days, table_days)
forecast_full = forecast_prices(scaled_close, MAX_DAYS)

forecast = forecast_full[:forecast_days]
forecast_table = forecast_full[:table_days]

future_dates_table = pd.date_range(
    start=recent_df["Date"].iloc[-1] + pd.Timedelta(days=1),
    periods=table_days
)

# ======================================================
# MARKET SNAPSHOT  ⬅️ ADD HERE
# ======================================================

st.markdown("<div class='section-title'>Market Outlook & Trade Signal</div>", unsafe_allow_html=True)

c0, c1, c2, c3 = st.columns(4)

# Market direction
direction = "Bullish" if forecast[0] > last_close else "Bearish"
dir_color = "positive" if direction == "Bullish" else "negative"
dir_arrow = "▲" if direction == "Bullish" else "▼"


h = forecast_days
horizon_label = f"{forecast_days} Day"

target_price = forecast[h - 1]
delta = target_price - last_close
pct = (delta / last_close) * 100
up = delta >= 0

arrow = "▲" if up else "▼"
sign = "+" if up else "-"
cls = "positive" if up else "negative"

with c0:
    st.markdown(f"""
    <div class="glass metric-card">
        <div class="metric-title">Last Closing Price</div>
        <div class="metric-value">${last_close:,.2f}</div>
        <div class="metric-sub">Previous market close</div>
    </div>
    """, unsafe_allow_html=True)
    
with c1:
    st.markdown(f"""
    <div class="glass metric-card" style="max-width:420px;">
        <div class="metric-title">{horizon_label} Forecast</div>
        <div class="metric-value">${target_price:,.2f}</div>
        <div class="metric-sub {cls}">
            {arrow} {sign}${abs(delta):.2f} ({sign}{abs(pct):.2f}%)
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="glass metric-card">
        <div class="metric-title">Market Bias</div>
        <div class="metric-value {dir_color}">{dir_arrow} {direction}</div>
        <div class="metric-sub">Based on next-day forecast</div>
    </div>
    """, unsafe_allow_html=True)





# Expected returns vs last close
forecast_returns = (forecast - last_close) / last_close

avg_return_pct = forecast_returns.mean() * 100

# Direction consistency
positive_days = np.sum(forecast_returns > 0)





# Trading signal
if avg_return_pct > 1:
    signal = "BUY"
    bias = "Bullish"
    color = "#22c55e"
    arrow = "▲"
elif avg_return_pct < -1:
    signal = "SELL"
    bias = "Bearish"
    color = "#ef4444"
    arrow = "▼"
else:
    signal = "HOLD"
    bias = "Neutral"
    color = "#facc15"
    arrow = "➖"

with c3:
    st.markdown(
        f"""
        <div class="glass metric-card" style="border-left:6px solid {color}">
            <div class="metric-title">Trading Signal</div>
            <div class="metric-value" style="color:{color}">
                {signal}
            </div>
            <div class="metric-sub">
                Avg Expected Return: <b>{avg_return_pct:.2f}%</b><br/>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ======================================================
# CANDLESTICK + FORECAST (SOLID CANDLES)
# ======================================================
st.markdown("<div class='section-title'>ecent Market Price Behavior</div>", unsafe_allow_html=True)

fig_candle = go.Figure()

fig_candle.add_trace(go.Candlestick(
    x=recent_df["Date"],
    open=recent_df["Open"],
    high=recent_df["High"],
    low=recent_df["Low"],
    close=recent_df["Close"],
    name="Price",
    increasing=dict(line=dict(color="#22c55e"), fillcolor="#22c55e"),
    decreasing=dict(line=dict(color="#ef4444"), fillcolor="#ef4444")
))

fig_candle.add_trace(go.Scatter(
    x=[recent_df["Date"].iloc[-1]] + list(future_dates),
    y=[last_close] + list(forecast),
    mode="lines+markers",
    name="Forecast",
    line=dict(color="#3b82f6", width=3, dash="dash"),
    marker=dict(size=6)
))

fig_candle.update_layout(
    template="plotly_dark",
    height=520,
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(fig_candle, width="stretch")

# ======================================================
# FORECAST OUTLOOK CARDS
# ======================================================


def outlook_card(title, price, base):
    delta = price - base
    pct = (delta / base) * 100
    up = delta >= 0
    arrow = "▲" if up else "▼"
    sign = "+" if up else "-"

    st.markdown(
        f"""
        <div class="glass metric-card" style="
            min-height: 150px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        ">
            <div class="metric-title">{title}</div>
            <div class="metric-value">${price:,.2f}</div>
            <div class="metric-sub {'positive' if up else 'negative'}">
                {arrow} {sign}${abs(delta):.2f} ({sign}{abs(pct):.2f}%)
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )




# ======================================================
# SHARPE RATIO (RISK-ADJUSTED METRIC)
# ======================================================

forecast_daily_returns = (
    np.diff(forecast_table) / forecast_table[:-1]
)
if len(forecast_daily_returns) > 1:
    std = forecast_daily_returns.std()

    if std != 0:
        sharpe_ratio = (
            forecast_daily_returns.mean() / std
        ) * np.sqrt(252)   # annualized
    else:
        sharpe_ratio = 0
else:
    sharpe_ratio = 0

# Sharpe interpretation
if sharpe_ratio > 1:
    sharpe_label = "Excellent Risk-Adjusted Return"
    sharpe_color = "#22c55e"
elif sharpe_ratio > 0.5:
    sharpe_label = "Moderate Risk-Adjusted Return"
    sharpe_color = "#facc15"
else:
    sharpe_label = "High Risk / Low Reward"
    sharpe_color = "#ef4444"




# ======================================================
# LINE PRICE CHART + FORECAST (CLEAN MARKET VIEW)
# ======================================================

fig_line = go.Figure()

# 1️⃣ Historical Close (SOLID)
fig_line.add_trace(go.Scatter(
    x=recent_df["Date"],
    y=recent_df["Close"],
    name="Historical Close",
    line=dict(color="#e5e7eb", width=2),
))

# 2️⃣ Forecast (CONNECTED, DASHED)
forecast_x = [recent_df["Date"].iloc[-1]] + list(future_dates)
forecast_y = [last_close] + list(forecast)

fig_line.add_trace(go.Scatter(
    x=forecast_x,
    y=forecast_y,
    mode="lines+markers",
    name="Forecast",
    line=dict(
        color="#3b82f6",
        width=3,
        dash="dash",
        shape="spline",
        smoothing=1.1
    ),
    marker=dict(size=6)
))

# Highlight forecast region
fig_line.add_vrect(
    x0=future_dates[0],
    x1=future_dates[-1],
    fillcolor="rgba(59,130,246,0.18)",
    layer="below",
    line_width=0
)

fig_line.update_layout(
    template="plotly_dark",
    height=420,
    hovermode="x unified",
    margin=dict(l=20, r=20, t=40, b=20),
    title="Historical vs Forecast Trend"
)

# ======================================================
# FORECAST CANDLESTICKS (PREDICTED ZONE)
# ======================================================

# Create synthetic forecast candles
forecast_open = [last_close] + list(forecast[:-1])
forecast_close = forecast

volatility = recent_df["Close"].pct_change().std()

forecast_high = forecast_close * (1 + volatility * 1.2)
forecast_low = forecast_close * (1 - volatility * 1.2)

fig_forecast_candle = go.Figure()

# Historical candles
fig_forecast_candle.add_trace(go.Candlestick(
    x=recent_df["Date"],
    open=recent_df["Open"],
    high=recent_df["High"],
    low=recent_df["Low"],
    close=recent_df["Close"],
    name="Historical",
    increasing=dict(fillcolor="#22c55e", line=dict(color="#22c55e")),
    decreasing=dict(fillcolor="#ef4444", line=dict(color="#ef4444"))
))

# Forecast candles
fig_forecast_candle.add_trace(go.Candlestick(
    x=future_dates,
    open=forecast_open,
    high=forecast_high,
    low=forecast_low,
    close=forecast_close,
    name="Forecast",
    opacity=1,
    increasing=dict(
        line=dict(color="#22c55e", width=2),
        fillcolor="rgba(34,197,94,0.12)"   # subtle green tint
    ),
    decreasing=dict(
        line=dict(color="#ef4444", width=2),
        fillcolor="rgba(239,68,68,0.12)"   # subtle red tint
    ),
    whiskerwidth=0.6
))

# Highlight forecast zone
fig_forecast_candle.add_vrect(
    x0=future_dates[0],
    x1=future_dates[-1],
    fillcolor="rgba(59,130,246,0.12)",
    layer="below",
    line_width=0
)

fig_forecast_candle.update_layout(
    template="plotly_dark",
    height=520,
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
    margin=dict(l=20, r=20, t=40, b=20)
)

COMMON_HEIGHT = 480

fig_line.update_layout(
    template="plotly_dark",
    height=COMMON_HEIGHT,
    hovermode="x unified",
    margin=dict(l=10, r=10, t=70, b=20),  # more top space for legend
    title="Price Trend View (Line)",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        font=dict(size=12)
    )
)

fig_forecast_candle.update_layout(
    template="plotly_dark",
    height=COMMON_HEIGHT,
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
    margin=dict(l=10, r=10, t=70, b=20),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        font=dict(size=12)
    )
)
st.markdown("<div class='section-title'>Forecast Outlook & Trend Analysis</div>", unsafe_allow_html=True)

col1, col2 = st.columns([1.1, 1.1])

with col1:
    st.plotly_chart(fig_forecast_candle, width="stretch")

with col2:
    st.plotly_chart(fig_line, width="stretch")





# ======================================================
# FORECAST TABLE (COLORED)
# ======================================================
rows = []
prev_price = last_close

for i in range(table_days):
    price = forecast_table[i]
    delta = price - prev_price
    pct = (delta / prev_price) * 100
    rows.append({
        "Day": i + 1,
        "Date": future_dates_table[i].date(),
        "Forecast Price ($)": round(price, 2),
        "Expected Change": f"{'▲' if delta >= 0 else '▼'} ${abs(delta):.2f}",
        "Change %": f"{'+' if delta >= 0 else '-'}{abs(pct):.2f}%"
    })
    prev_price = price

forecast_df = pd.DataFrame(rows)

def color_change(val):
    if "▲" in val or "+" in val:
        return "color:#22c55e;font-weight:600;"
    if "▼" in val or "-" in val:
        return "color:#ef4444;font-weight:600;"
    return ""

styled_df = forecast_df.style.map(color_change, subset=["Expected Change", "Change %"])



# =========================
# MAIN SPLIT: LEFT / RIGHT
# =========================
st.markdown("<div class='section-title'>Forecast Breakdown & Market Outlook</div>", unsafe_allow_html=True)

left_col, right_col = st.columns([1.4, 1])

# -------------------------------------------------
# LEFT: FORECAST TABLE
# -------------------------------------------------
with left_col:
    if forecast_days < 10:
        st.caption(
            "ℹ️ A minimum 10-day forecast is displayed for the table and summary cards "
            "to ensure consistent insights."
        )
    st.dataframe(styled_df, width="stretch")

# -------------------------------------------------
# RIGHT: 2 ROWS × 2 COLUMNS (OUTLOOK + SHARPE)
# -------------------------------------------------
with right_col:
    st.markdown("<div style='height:5vh'></div>", unsafe_allow_html=True)
    # ---------- ROW 1 ----------
    r1 = st.container()
    with r1:
        c1, c2 = st.columns(2, gap="large")

        with c1:
            outlook_card("1-Day Forecast", forecast_table[0], last_close)

        with c2:
            outlook_card("5-Day Forecast", forecast_table[4], last_close)

  
    st.markdown("<div style='height:5vh'></div>", unsafe_allow_html=True)
    # ---------- ROW 2 ----------
    r2 = st.container()
    with r2:
        c3, c4 = st.columns(2, gap="large")

        with c3:
            outlook_card("10-Day Forecast", forecast_table[9], last_close)
        with c4:
            st.markdown(
    f"""
    <div class="glass metric-card" style="
        border-left:6px solid {sharpe_color};
        min-height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    ">
        <div class="metric-title">Sharpe Ratio</div>
        <div class="metric-value" style="color:{sharpe_color}">
            {sharpe_ratio:.2f}
        </div>
        <div class="metric-sub">
            {sharpe_label}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ======================================================
# RISK & MARKET ANALYSIS (LAST 60 DAYS)
# ======================================================
st.markdown("<div class='section-title'>Risk & Market Analysis</div>", unsafe_allow_html=True)

recent_df["Daily Return %"] = recent_df["Close"].pct_change() * 100
recent_df["Volatility"] = recent_df["Daily Return %"].rolling(10).std()

colors = ["#22c55e" if r >= 0 else "#ef4444" for r in recent_df["Daily Return %"]]

fig_returns = go.Figure()

fig_returns.add_trace(go.Bar(
    x=recent_df["Date"],
    y=recent_df["Daily Return %"],
    marker_color=colors,
    opacity=0.85
))

# Zero baseline (VERY important for market UX)
fig_returns.add_hline(
    y=0,
    line_width=1,
    line_dash="dot",
    line_color="rgba(255,255,255,0.4)"
)

fig_returns.update_layout(
    template="plotly_dark",
    height=300,
    title=dict(
        text="Daily Returns (Last 60 Days)",
        x=0.01,
        y=0.92
    ),
    margin=dict(l=40, r=30, t=50, b=40),
    bargap=0.25,
    hovermode="x unified",
    xaxis=dict(
        showgrid=False
    ),
    yaxis=dict(
        title="Return (%)",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.05)"
    )
)
fig_vol = go.Figure()

fig_vol.add_trace(go.Scatter(
    x=recent_df["Date"],
    y=recent_df["Volatility"],
    mode="lines",
    line=dict(color="#facc15", width=2.5, shape="spline", smoothing=1.1),
    fill="tozeroy",
    fillcolor="rgba(250,204,21,0.15)",
    name="Volatility"
))

fig_vol.update_layout(
    template="plotly_dark",
    height=300,
    title=dict(
        text="10-Day Rolling Volatility",
        x=0.01,
        y=0.92
    ),
    margin=dict(l=40, r=30, t=50, b=40),
    hovermode="x unified",
    xaxis=dict(
        showgrid=False
    ),
    yaxis=dict(
        title="Volatility",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.05)"
    )
)
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_returns, width="stretch")
with col2:
    st.plotly_chart(fig_vol, width="stretch")





# ======================================================
# MULTI-MODEL FORECAST OVERLAY (CORRECT & COMPLETE)
# ======================================================

if show_model_comparison:

    st.markdown("<div class='section-title'>Multi-Model Forecast Comparison</div>", unsafe_allow_html=True)

    fig_multi = go.Figure()

    # =============================
    # Historical line
    # =============================
    fig_multi.add_trace(go.Scatter(
        x=recent_df["Date"],
        y=recent_df["Close"],
        name="Historical",
        line=dict(color="#9ca3af", width=2),
        opacity=0.85
    ))

    # =============================
    # Model styles
    # =============================
    model_styles = {
        "LSTM_Tuned": ("#22c55e", "solid"),
        "LSTM_Baseline": ("#10b981", "dot"),
        "SimpleRNN_Tuned": ("#3b82f6", "solid"),
        "SimpleRNN_Baseline": ("#60a5fa", "dot"),
    }

    # =============================
    # Forecast for each model
    # =============================
    for model_name, model_file in model_files.items():

        temp_model = load_model(MODEL_DIR / model_file, compile=False)
        

        X = scaled_close[-LOOKBACK:].reshape(1, LOOKBACK, 1)

        preds_scaled = temp_model.predict(X, verbose=0).flatten()
        preds_scaled = preds_scaled[:forecast_days].reshape(-1, 1)

        preds_real = scaler.inverse_transform(preds_scaled).flatten()

        plot_x = [recent_df["Date"].iloc[-1]] + list(future_dates)
        plot_y = [last_close] + list(preds_real)

        color, dash_style = model_styles[model_name]

        fig_multi.add_trace(go.Scatter(
            x=plot_x,
            y=plot_y,
            name=model_name,
            mode="lines",
            line=dict(
                color=color,
                width=2.5,
                dash=dash_style,
                shape="spline",
                smoothing=1.1
            ),
            opacity=0.9
        ))

    # =============================
    # Forecast start marker
    # =============================
    forecast_start_date = recent_df["Date"].iloc[-1].to_pydatetime()

    fig_multi.add_vline(
        x=forecast_start_date,
        line_width=1,
        line_dash="dot",
        line_color="#facc15"
    )

    fig_multi.add_annotation(
        x=forecast_start_date,
        y=max(recent_df["Close"]) * 1.03,
        text="Forecast Start",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#facc15",
        font=dict(color="#facc15", size=12),
        yanchor="bottom"
    )

    # =============================
    # Layout (Spacing Optimized)
    # =============================
    fig_multi.update_layout(
        template="plotly_dark",
        height=460,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.10,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)"
        )
    )

    st.plotly_chart(fig_multi, width="stretch")


# ==============================
# TEXT INTERPRETATION VARIABLES
# ==============================

# Trend direction
trend_text = (
    "upward" if avg_return_pct > 0 else
    "downward" if avg_return_pct < 0 else
    "sideways"
)

# Confidence level (human readable)
if confidence_score is None:
    confidence_text = "unknown"
else:
    confidence_text = (
        "high" if confidence_score >= 65 else
        "moderate" if confidence_score >= 40 else
        "low"
    )

decision_text = (
    "BUY" if avg_return_pct > 1 and sharpe_ratio > 0
    else "SELL" if avg_return_pct < -1 and sharpe_ratio < 0
    else "HOLD"
)

decision_reason = (
    "expected returns are positive and risk-adjusted performance is favorable"
    if decision_text == "BUY"
    else "expected returns are negative and risk-adjusted performance is weak"
    if decision_text == "SELL"
    else "signals are mixed and do not strongly favor either direction"
)
st.markdown("<div class='section-title'>Forecast Interpretation & Recommendation</div>", unsafe_allow_html=True)


st.markdown(
    f"""
    <div class="glass metric-card" style="
        width:100%;
        padding:28px 36px;
        margin-bottom:18px;
    ">
        <div class="metric-sub" style="line-height:1.8; font-size:15px;">
            This forecast represents the model’s best estimate of Tesla’s future
            price direction based on recent historical market behavior.
            <br/><br/>
            The model does not attempt to predict exact future prices. Instead,
            it learns statistically recurring patterns from past data and
            projects how price may evolve over the selected forecast horizon.
            Results should be interpreted as directional guidance rather than
            guaranteed outcomes.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    f"""
    <div class="glass metric-card" style="
        width:100%;
        padding:28px 36px;
        margin-bottom:18px;
    ">
        <div class="metric-sub" style="line-height:1.9; font-size:15px;">
            Forecast summary based on the selected horizon:
            <br/><br/>
            • Expected average return: <b>{avg_return_pct:.2f}%</b><br/>
            • Market bias indicated by the model: <b>{bias}</b><br/>
            • Forecast confidence level: <b>{confidence_text.capitalize()}</b><br/>
            • Risk-adjusted performance (Sharpe Ratio): <b>{sharpe_ratio:.2f}</b>
            <br/><br/>
            Based on these signals, the model suggests a
            <b>{decision_text}</b> stance because
            {decision_reason}.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <div class="glass metric-card" style="
        width:100%;
        padding:24px 36px;
        opacity:0.85;
    ">
        <div class="metric-sub" style="line-height:1.7; font-size:14px;">
            NOTE: This analysis should be interpreted as a directional guide rather
            than a precise price target. Forecast reliability may decrease during
            periods of elevated volatility or unexpected market events.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="app-footer">
        <span class="footer-warning">⚠️ Educational Use Only</span><br/>
        <span class="footer-muted">
            This dashboard is intended for learning and analytical purposes.
            It does not constitute
            <span class="footer-highlight">financial advice</span> or
            trading recommendations.
        </span>
    </div>
    """,
    unsafe_allow_html=True
)
