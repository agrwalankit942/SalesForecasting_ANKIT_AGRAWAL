"""
Sales Forecasting & Demand Intelligence System — Streamlit Dashboard
=====================================================================
Run with:  streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import joblib # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import plotly.express as px # type: ignore
import plotly.graph_objects as go # type: ignore
import streamlit as st # type: ignore
from sklearn.ensemble import IsolationForest # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error # type: ignore

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sales Forecasting & Demand Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# DATA / MODEL LOADING (cached)
# ----------------------------------------------------------------------------
DATA_PATH = "clean_sales_data.csv"
SEGMENTS_PATH = "Product_Demand_Segments.csv"


@st.cache_data(show_spinner="Loading sales data...")
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    if "Ship Date" in df.columns:
        df["Ship Date"] = pd.to_datetime(df["Ship Date"])
    return df


@st.cache_data(show_spinner="Loading product demand segments...")
def load_segments():
    return pd.read_csv(SEGMENTS_PATH)


@st.cache_resource(show_spinner="Loading trained models...")
def load_models():
    models = {}
    try:
        models["prophet"] = joblib.load("prophet_model.pkl")
    except Exception:
        models["prophet"] = None
    try:
        models["xgb"] = joblib.load("xgb_model.pkl")
    except Exception:
        models["xgb"] = None
    try:
        models["kmeans"] = joblib.load("kmeans_model.pkl")
    except Exception:
        models["kmeans"] = None
    try:
        models["pca"] = joblib.load("pca_model.pkl")
    except Exception:
        models["pca"] = None
    try:
        models["scaler"] = joblib.load("scaler.pkl")
    except Exception:
        models["scaler"] = None
    return models


df = load_data()
segments_df = load_segments()
models = load_models()

# ----------------------------------------------------------------------------
# SHARED HELPERS
# ----------------------------------------------------------------------------
SEASON_MAP_MONTH = {12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4}


def get_monthly_series(data, filter_col=None, filter_val=None):
    """Aggregate Sales to month-end totals, optionally filtered by a column value."""
    d = data
    if filter_col and filter_val and filter_val != "Overall (All Sales)":
        d = d[d[filter_col] == filter_val]
    monthly = (
        d.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"]
        .sum()
        .reset_index()
    )
    monthly.columns = ["ds", "y"]
    return monthly


@st.cache_data(show_spinner=False)
def build_xgb_features(monthly):
    """Build lag / rolling / calendar features exactly as the training notebook did."""
    xdf = monthly.copy()
    xdf["Lag_1"] = xdf["y"].shift(1)
    xdf["Lag_2"] = xdf["y"].shift(2)
    xdf["Lag_3"] = xdf["y"].shift(3)
    xdf["Rolling_Mean_3"] = xdf["y"].rolling(window=3).mean()
    xdf["Month"] = xdf["ds"].dt.month
    xdf["Quarter"] = xdf["ds"].dt.quarter
    xdf["Season"] = xdf["Month"].map(SEASON_MAP_MONTH)
    xdf = xdf.dropna().reset_index(drop=True)
    return xdf


FEATURES = ["Lag_1", "Lag_2", "Lag_3", "Rolling_Mean_3", "Month", "Quarter", "Season"]


@st.cache_data(show_spinner=False)
def run_prophet(monthly, periods=3):
    """Train/test split evaluation + full-data future forecast (mirrors the notebook)."""
    from prophet import Prophet # type: ignore

    if len(monthly) < 8:
        return None

    train, test = monthly.iloc[:-periods], monthly.iloc[-periods:]

    m_eval = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                      daily_seasonality=False, seasonality_mode="additive")
    m_eval.fit(train)
    future_test = m_eval.make_future_dataframe(periods=periods, freq="ME")
    fc_test = m_eval.predict(future_test)
    preds = fc_test["yhat"].tail(periods).values
    actual = test["y"].values
    mae = mean_absolute_error(actual, preds)
    rmse = np.sqrt(mean_squared_error(actual, preds))

    m_final = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                       daily_seasonality=False, seasonality_mode="additive")
    m_final.fit(monthly)
    future = m_final.make_future_dataframe(periods=periods, freq="ME")
    forecast = m_final.predict(future)
    future_forecast = forecast[["ds", "yhat"]].tail(periods).reset_index(drop=True)

    return {"model": "Prophet", "mae": mae, "rmse": rmse, "future": future_forecast}


@st.cache_data(show_spinner=False)
def run_xgb(monthly, periods=3):
    """Train/test split evaluation + recursive future forecast (mirrors the notebook)."""
    from xgboost import XGBRegressor # type: ignore

    xdf = build_xgb_features(monthly)
    if len(xdf) < periods + 5:
        return None

    X, y = xdf[FEATURES], xdf["y"]
    X_train, X_test = X.iloc[:-periods], X.iloc[-periods:]
    y_train, y_test = y.iloc[:-periods], y.iloc[-periods:]

    model_eval = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
    model_eval.fit(X_train, y_train)
    preds = model_eval.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    model_final = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
    model_final.fit(X, y)

    future_rows = []
    last_row = xdf.iloc[-1:].copy()
    cursor_date = last_row["ds"].values[0]
    for _ in range(periods):
        X_future = last_row[FEATURES]
        pred = float(model_final.predict(X_future)[0])
        cursor_date = pd.Timestamp(cursor_date) + pd.offsets.MonthEnd(1)
        future_rows.append({"ds": cursor_date, "yhat": pred})

        new_lag3 = last_row["Lag_2"].values[0]
        new_lag2 = last_row["Lag_1"].values[0]
        new_lag1 = pred
        last_row = last_row.copy()
        last_row["Lag_3"] = new_lag3
        last_row["Lag_2"] = new_lag2
        last_row["Lag_1"] = new_lag1
        last_row["Rolling_Mean_3"] = np.mean([new_lag1, new_lag2, new_lag3])
        last_row["ds"] = cursor_date
        last_row["Month"] = cursor_date.month
        last_row["Quarter"] = cursor_date.quarter
        last_row["Season"] = SEASON_MAP_MONTH[cursor_date.month]

    future_forecast = pd.DataFrame(future_rows)
    return {"model": "XGBoost", "mae": mae, "rmse": rmse, "future": future_forecast}


@st.cache_data(show_spinner="Training Prophet & XGBoost for this selection...")
def get_best_forecast(_data_signature, filter_col, filter_val, periods=3):
    """
    Trains BOTH Prophet and XGBoost for the chosen slice of data, evaluates both
    on a 3-month holdout, and returns whichever has the lower RMSE (matching the
    'Best Model Selection' logic used in the notebook), plus both results for reference.
    """
    monthly = get_monthly_series(df, filter_col, filter_val)
    prophet_res = run_prophet(monthly, periods=3)
    xgb_res = run_xgb(monthly, periods=3)

    candidates = [r for r in [prophet_res, xgb_res] if r is not None]
    if not candidates:
        return None, monthly, prophet_res, xgb_res

    best = min(candidates, key=lambda r: r["rmse"])
    return best, monthly, prophet_res, xgb_res


# ----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "1️⃣ Sales Overview",
        "2️⃣ Forecast Explorer",
        "3️⃣ Anomaly Report",
        "4️⃣ Product Demand Segments",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Sales Forecasting & Demand Intelligence System\n\n"
    "Models: SARIMA · Prophet · XGBoost\n\n"
    f"Data: {df['Order Date'].min().date()} → {df['Order Date'].max().date()}"
)

# ============================================================================
# PAGE 1 — SALES OVERVIEW DASHBOARD
# ============================================================================
if page.startswith("1"):
    st.title("📈 Sales Overview Dashboard")
    st.caption("High-level view of historical sales performance.")

    with st.container():
        c1, c2, c3 = st.columns(3)
        regions = ["All"] + sorted(df["Region"].unique().tolist())
        categories = ["All"] + sorted(df["Category"].unique().tolist())
        years = ["All"] + sorted(df["Year"].unique().tolist())
        sel_region = c1.selectbox("Region", regions)
        sel_category = c2.selectbox("Category", categories)
        sel_year = c3.selectbox("Year", years)

    filtered = df.copy()
    if sel_region != "All":
        filtered = filtered[filtered["Region"] == sel_region]
    if sel_category != "All":
        filtered = filtered[filtered["Category"] == sel_category]
    if sel_year != "All":
        filtered = filtered[filtered["Year"] == sel_year]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Sales", f"${filtered['Sales'].sum():,.0f}")
    k2.metric("Total Orders", f"{len(filtered):,}")
    k3.metric("Avg Order Value", f"${filtered['Sales'].mean():,.2f}" if len(filtered) else "—")
    k4.metric("Unique Products", f"{filtered['Product Name'].nunique():,}")

    st.markdown("### Total Sales by Year")
    yearly = filtered.groupby("Year")["Sales"].sum().reset_index()
    fig_year = px.bar(
        yearly, x="Year", y="Sales", text_auto=".2s",
        color="Sales", color_continuous_scale="Blues",
    )
    fig_year.update_layout(yaxis_title="Total Sales ($)", showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig_year, use_container_width=True)

    st.markdown("### Monthly Sales Trend")
    monthly_trend = (
        filtered.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"]
        .sum()
        .reset_index()
    )
    fig_trend = px.line(monthly_trend, x="Order Date", y="Sales", markers=True)
    fig_trend.update_layout(yaxis_title="Sales ($)", xaxis_title="Month")
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("### Sales by Region & Category")
    c1, c2 = st.columns(2)
    with c1:
        by_region = filtered.groupby("Region")["Sales"].sum().reset_index().sort_values("Sales", ascending=False)
        fig_region = px.bar(by_region, x="Region", y="Sales", color="Region", text_auto=".2s")
        fig_region.update_layout(showlegend=False, yaxis_title="Sales ($)")
        st.plotly_chart(fig_region, use_container_width=True)
    with c2:
        by_cat = filtered.groupby("Category")["Sales"].sum().reset_index().sort_values("Sales", ascending=False)
        fig_cat = px.bar(by_cat, x="Category", y="Sales", color="Category", text_auto=".2s")
        fig_cat.update_layout(showlegend=False, yaxis_title="Sales ($)")
        st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown("### Region × Category Breakdown")
    pivot = filtered.pivot_table(index="Region", columns="Category", values="Sales", aggfunc="sum", fill_value=0)
    fig_heat = px.imshow(pivot, text_auto=".2s", color_continuous_scale="Blues", aspect="auto")
    st.plotly_chart(fig_heat, use_container_width=True)

# ============================================================================
# PAGE 2 — FORECAST EXPLORER
# ============================================================================
elif page.startswith("2"):
    st.title("🔮 Forecast Explorer")
    st.caption("Forecasts are generated live using both Prophet and XGBoost; the model with the lower RMSE on a 3-month holdout is shown as the 'best' forecast.")

    c1, c2 = st.columns(2)
    with c1:
        level = st.selectbox("Forecast by", ["Overall (All Sales)", "Category", "Region"])
    with c2:
        if level == "Category":
            value = st.selectbox("Select Category", sorted(df["Category"].unique()))
            filter_col = "Category"
        elif level == "Region":
            value = st.selectbox("Select Region", sorted(df["Region"].unique()))
            filter_col = "Region"
        else:
            value = "Overall (All Sales)"
            filter_col = None

    horizon = st.select_slider(
        "Forecast horizon (months ahead)",
        options=[1, 2, 3],
        value=3,
    )

    best, monthly, prophet_res, xgb_res = get_best_forecast(
        f"{filter_col}-{value}", filter_col, value, periods=3
    )

    if best is None:
        st.warning("Not enough historical data to generate a forecast for this selection.")
    else:
        future_view = best["future"].head(horizon)

        st.success(f"**Best Model for this selection: {best['model']}** (lowest RMSE on holdout test)")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["ds"], y=monthly["y"], mode="lines", name="Historical Sales",
            line=dict(color="#4C78A8", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=future_view["ds"], y=future_view["yhat"], mode="lines+markers", name=f"{best['model']} Forecast",
            line=dict(color="#E45756", width=3, dash="dash"), marker=dict(size=9),
        ))
        fig.update_layout(
            title=f"{value} — {horizon}-Month Sales Forecast ({best['model']})",
            xaxis_title="Date", yaxis_title="Sales ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Forecast Values")
        table = future_view.rename(columns={"ds": "Month", "yhat": "Forecasted Sales"}).copy()
        table["Month"] = table["Month"].dt.strftime("%B %Y")
        table["Forecasted Sales"] = table["Forecasted Sales"].round(2)
        st.dataframe(table, use_container_width=True, hide_index=True)

        st.markdown("### Model Performance (3-month holdout)")
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{best['model']} MAE", f"${best['mae']:,.2f}")
        m2.metric(f"{best['model']} RMSE", f"${best['rmse']:,.2f}")
        m3.metric("Model Used", best["model"])

        with st.expander("Compare Prophet vs XGBoost for this selection"):
            comp_rows = []
            if prophet_res:
                comp_rows.append({"Model": "Prophet", "MAE": round(prophet_res["mae"], 2), "RMSE": round(prophet_res["rmse"], 2)})
            if xgb_res:
                comp_rows.append({"Model": "XGBoost", "MAE": round(xgb_res["mae"], 2), "RMSE": round(xgb_res["rmse"], 2)})
            comp_df = pd.DataFrame(comp_rows)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            fig_comp = px.bar(comp_df.melt(id_vars="Model", var_name="Metric", value_name="Value"),
                               x="Model", y="Value", color="Metric", barmode="group")
            st.plotly_chart(fig_comp, use_container_width=True)

# ============================================================================
# PAGE 3 — ANOMALY REPORT
# ============================================================================
elif page.startswith("3"):
    st.title("🚨 Anomaly Report")
    st.caption("Weekly sales anomalies detected using Isolation Forest (Task 5).")

    @st.cache_data(show_spinner="Detecting anomalies...")
    def detect_anomalies(_sig):
        weekly = df.groupby(pd.Grouper(key="Order Date", freq="W"))["Sales"].sum().reset_index()
        iso = IsolationForest(contamination=0.05, random_state=42)
        weekly["IF_Anomaly"] = iso.fit_predict(weekly[["Sales"]])
        weekly["IF_Anomaly"] = weekly["IF_Anomaly"].map({1: "Normal", -1: "Anomaly"})
        return weekly

    weekly_sales = detect_anomalies("weekly")
    anomalies = weekly_sales[weekly_sales["IF_Anomaly"] == "Anomaly"]

    k1, k2 = st.columns(2)
    k1.metric("Weeks Analyzed", f"{len(weekly_sales):,}")
    k2.metric("Anomalies Detected", f"{len(anomalies):,}")

    st.markdown("### Weekly Sales — Anomaly Chart")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weekly_sales["Order Date"], y=weekly_sales["Sales"], mode="lines",
        name="Weekly Sales", line=dict(color="#4C78A8", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=anomalies["Order Date"], y=anomalies["Sales"], mode="markers",
        name="Anomaly", marker=dict(color="red", size=11, symbol="circle"),
    ))
    fig.update_layout(
        title="Isolation Forest — Weekly Sales Anomalies",
        xaxis_title="Date", yaxis_title="Sales ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Detected Anomaly Dates")
    anomaly_table = anomalies[["Order Date", "Sales"]].copy()
    anomaly_table.columns = ["Week Ending", "Sales ($)"]
    anomaly_table["Week Ending"] = anomaly_table["Week Ending"].dt.strftime("%Y-%m-%d")
    anomaly_table["Sales ($)"] = anomaly_table["Sales ($)"].round(2)

    def explain(week_ending):
        month = pd.to_datetime(week_ending).strftime("%B")
        if month in ["November", "December"]:
            return "Holiday season / Black Friday / Christmas sales"
        elif month == "January":
            return "Post-holiday demand drop"
        return "Promotion, supply disruption, or unexpected demand"

    anomaly_table["Possible Reason"] = anomaly_table["Week Ending"].apply(explain)
    st.dataframe(anomaly_table, use_container_width=True, hide_index=True)

# ============================================================================
# PAGE 4 — PRODUCT DEMAND SEGMENTS
# ============================================================================
elif page.startswith("4"):
    st.title("🧩 Product Demand Segments")
    st.caption("Sub-categories clustered by Total Sales, Growth Rate, Volatility & Average Order Value (K-Means, Task 6).")

    st.markdown("### Cluster Map (PCA 2D Projection)")
    fig = px.scatter(
        segments_df, x="PC1", y="PC2", color="Demand Segment", text="Sub-Category",
        size="TotalSales", size_max=40, hover_data=["TotalSales", "GrowthRate", "Volatility", "AverageOrderValue"],
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(
        title="Product Demand Segments",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Sub-Categories by Demand Segment")
    for seg in sorted(segments_df["Demand Segment"].unique()):
        with st.expander(f"📦 {seg}", expanded=True):
            seg_table = segments_df[segments_df["Demand Segment"] == seg][
                ["Sub-Category", "TotalSales", "GrowthRate", "Volatility", "AverageOrderValue"]
            ].sort_values("TotalSales", ascending=False).reset_index(drop=True)
            seg_table["TotalSales"] = seg_table["TotalSales"].round(2)
            seg_table["GrowthRate"] = (seg_table["GrowthRate"] * 100).round(1).astype(str) + "%"
            seg_table["Volatility"] = seg_table["Volatility"].round(2)
            seg_table["AverageOrderValue"] = seg_table["AverageOrderValue"].round(2)
            st.dataframe(seg_table, use_container_width=True, hide_index=True)

    st.markdown("### Full Segment Table")
    full_table = segments_df[["Sub-Category", "TotalSales", "GrowthRate", "Volatility", "AverageOrderValue", "Demand Segment"]]
    st.dataframe(full_table, use_container_width=True, hide_index=True)