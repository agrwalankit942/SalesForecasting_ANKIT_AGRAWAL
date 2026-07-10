"""
===========================================================
Sales Forecasting & Demand Intelligence Dashboard
Author : Ankit Agrawal
===========================================================
"""

# -------------------------------------------------------------
# THREAD / OPENMP SAFETY
# -------------------------------------------------------------
# Prophet (via cmdstanpy) and XGBoost both ship their own vendored
# OpenMP runtime. Loading two different OpenMP runtimes into the
# same process is a well known cause of native (non-Python)
# crashes such as "Segmentation fault" on constrained containers
# like Streamlit Community Cloud. Pinning thread counts to 1 and
# allowing duplicate runtimes to coexist avoids that class of crash.
# These MUST be set before numpy / xgboost / prophet are imported.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

import streamlit as st # type: ignore

from sklearn.metrics import mean_absolute_error, mean_squared_error

# NOTE: Prophet, XGBRegressor and IsolationForest are intentionally
# imported lazily (inside the functions/pages that use them) instead
# of at the top of the module. Prophet's import pulls in cmdstanpy,
# which is one of the heaviest and most crash-prone imports in this
# stack on Streamlit Cloud's memory-limited containers. Importing it
# only when the user actually opens "Forecast Explorer" or "Anomaly
# Report" keeps the app's default landing page (Sales Overview)
# lightweight and avoids paying that cost - and that risk - on
# every single run.

# -------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(
    page_title="Sales Forecasting & Demand Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# FILE PATHS
# -------------------------------------------------------------
DATA_PATH = "clean_sales_data.csv"
SEGMENT_PATH = "Product_Demand_Segments.csv"

# -------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------
@st.cache_data
def load_data():

    if not os.path.exists(DATA_PATH):
        return None

    df = pd.read_csv(DATA_PATH)

    df["Order Date"] = pd.to_datetime(df["Order Date"])

    if "Ship Date" in df.columns:
        df["Ship Date"] = pd.to_datetime(df["Ship Date"])

    return df


@st.cache_data
def load_segments():

    if not os.path.exists(SEGMENT_PATH):
        return pd.DataFrame()

    return pd.read_csv(SEGMENT_PATH)


df = load_data()
segments_df = load_segments()

if df is None:
    st.error(
        f"⚠️ Required data file '{DATA_PATH}' was not found in the "
        "repository. Please make sure it is committed alongside app.py."
    )
    st.stop()

# -------------------------------------------------------------
# LOAD OTHER MODELS
# -------------------------------------------------------------
# NOTE: kmeans_model.pkl / pca_model.pkl / scaler.pkl were being
# unpickled here via joblib.load() at every app startup, but the
# resulting objects were never referenced anywhere else in this
# file (the Product Demand Segments page reads the pre-computed
# Product_Demand_Segments.csv instead). Unpickling sklearn/numpy
# objects that were trained with a different numpy/scikit-learn/
# joblib version than what's pinned in requirements.txt is a
# common cause of a hard, non-Python-catchable crash (segfault)
# during unpickling. Since these objects were unused dead weight,
# the loading has been removed rather than risk that crash.
#
# If you need these models again later, re-export them with
# joblib using the exact scikit-learn/numpy versions pinned in
# requirements.txt, then reintroduce a *lazy* loader (called only
# from the page that needs it) wrapped in try/except.

# -------------------------------------------------------------
# HELPER CONSTANTS
# -------------------------------------------------------------
SEASON_MAP = {

    12:1,
    1:1,
    2:1,

    3:2,
    4:2,
    5:2,

    6:3,
    7:3,
    8:3,

    9:4,
    10:4,
    11:4
}

# -------------------------------------------------------------
# MONTHLY SALES
# -------------------------------------------------------------
def monthly_sales(data,
                  filter_col=None,
                  filter_value=None):

    temp = data.copy()

    if filter_col is not None:

        temp = temp[temp[filter_col] == filter_value]

    monthly = (

        temp.groupby(
            pd.Grouper(
                key="Order Date",
                freq="ME"
            )
        )["Sales"]

        .sum()

        .reset_index()

    )

    monthly.columns = ["ds","y"]

    return monthly

# -------------------------------------------------------------
# XGBOOST FEATURES
# -------------------------------------------------------------
def create_features(monthly):

    df_feat = monthly.copy()

    df_feat["Lag1"] = df_feat["y"].shift(1)
    df_feat["Lag2"] = df_feat["y"].shift(2)
    df_feat["Lag3"] = df_feat["y"].shift(3)

    df_feat["RollingMean"] = (

        df_feat["y"]
        .rolling(3)
        .mean()

    )

    df_feat["Month"] = df_feat["ds"].dt.month

    df_feat["Quarter"] = df_feat["ds"].dt.quarter

    df_feat["Season"] = df_feat["Month"].map(SEASON_MAP)

    df_feat = df_feat.dropna()

    return df_feat

FEATURES = [

    "Lag1",
    "Lag2",
    "Lag3",
    "RollingMean",
    "Month",
    "Quarter",
    "Season"

]

# -------------------------------------------------------------
# PROPHET FORECAST
# -------------------------------------------------------------
def prophet_forecast(monthly,
                     periods=3):

    if len(monthly) < 8:

        return None

    # Lazy import: keeps Prophet/cmdstanpy out of the process unless
    # this function actually runs (see note at top of file).
    try:
        from prophet import Prophet
    except Exception as e:
        st.warning(f"Prophet is unavailable in this environment: {e}")
        return None

    train = monthly.iloc[:-periods]

    test = monthly.iloc[-periods:]

    model = Prophet(

        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False

    )

    model.fit(train)

    future = model.make_future_dataframe(

        periods=periods,
        freq="ME"

    )

    forecast = model.predict(future)

    pred = forecast["yhat"].tail(periods).values

    mae = mean_absolute_error(

        test["y"],
        pred

    )

    rmse = np.sqrt(

        mean_squared_error(

            test["y"],
            pred

        )

    )

    final_model = Prophet(

        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False

    )

    final_model.fit(monthly)

    future = final_model.make_future_dataframe(

        periods=periods,
        freq="ME"

    )

    forecast = final_model.predict(future)

    future = forecast[["ds","yhat"]].tail(periods)

    return {

        "model":"Prophet",

        "forecast":future,

        "mae":mae,

        "rmse":rmse

    }

# -------------------------------------------------------------
# XGBOOST FORECAST
# -------------------------------------------------------------
def xgb_forecast(monthly,
                 periods=3):

    # Lazy import: see note at top of file.
    from xgboost import XGBRegressor

    feat = create_features(monthly)

    if len(feat) < 8:

        return None

    X = feat[FEATURES]

    y = feat["y"]

    X_train = X.iloc[:-periods]

    X_test = X.iloc[-periods:]

    y_train = y.iloc[:-periods]

    y_test = y.iloc[-periods:]

    model = XGBRegressor(

        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
        n_jobs=1

    )

    model.fit(

        X_train,
        y_train

    )

    pred = model.predict(X_test)

    mae = mean_absolute_error(

        y_test,
        pred

    )

    rmse = np.sqrt(

        mean_squared_error(

            y_test,
            pred

        )

    )

    final = XGBRegressor(

        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
        n_jobs=1

    )

    final.fit(X,y)

    last = feat.iloc[-1:].copy()

    future_rows = []

    current = last["ds"].iloc[0]

    for i in range(periods):

        prediction = float(

            final.predict(
                last[FEATURES]
            )[0]

        )

        current = current + pd.offsets.MonthEnd()

        future_rows.append({

            "ds":current,

            "yhat":prediction

        })

        lag3 = last["Lag2"].iloc[0]

        lag2 = last["Lag1"].iloc[0]

        lag1 = prediction

        last["Lag3"] = lag3
        last["Lag2"] = lag2
        last["Lag1"] = lag1

        last["RollingMean"] = np.mean([

            lag1,
            lag2,
            lag3

        ])

        last["Month"] = current.month

        last["Quarter"] = current.quarter

        last["Season"] = SEASON_MAP[current.month]

        last["ds"] = current

    future = pd.DataFrame(future_rows)

    return {

        "model":"XGBoost",

        "forecast":future,

        "mae":mae,

        "rmse":rmse

    }

# -------------------------------------------------------------
# BEST MODEL
# -------------------------------------------------------------
def best_model(monthly,
               periods=3):

    prophet = prophet_forecast(

        monthly,
        periods

    )

    xgb = xgb_forecast(

        monthly,
        periods

    )

    models = []

    if prophet is not None:
        models.append(prophet)

    if xgb is not None:
        models.append(xgb)

    if len(models)==0:
        return None

    best = min(

        models,

        key=lambda x:x["rmse"]

    )

    return best, prophet, xgb
# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("📊 Dashboard Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "📈 Sales Overview",
        "🔮 Forecast Explorer",
        "🚨 Anomaly Report",
        "📦 Product Demand Segments"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
**Sales Forecasting & Demand Intelligence**

Models Used

✅ Prophet

✅ XGBoost

Machine Learning

✅ Isolation Forest

✅ KMeans Clustering

"""
)

# ============================================================
# PAGE 1 : SALES OVERVIEW
# ============================================================

if page == "📈 Sales Overview":

    st.title("📈 Sales Overview Dashboard")

    st.markdown("### Interactive Filters")

    c1, c2, c3 = st.columns(3)

    with c1:

        regions = ["All"] + sorted(df["Region"].unique())

        selected_region = st.selectbox(
            "Region",
            regions
        )

    with c2:

        categories = ["All"] + sorted(df["Category"].unique())

        selected_category = st.selectbox(
            "Category",
            categories
        )

    with c3:

        years = ["All"] + sorted(df["Year"].unique())

        selected_year = st.selectbox(
            "Year",
            years
        )

    filtered = df.copy()

    if selected_region != "All":

        filtered = filtered[
            filtered["Region"] == selected_region
        ]

    if selected_category != "All":

        filtered = filtered[
            filtered["Category"] == selected_category
        ]

    if selected_year != "All":

        filtered = filtered[
            filtered["Year"] == selected_year
        ]

    # ------------------------------------------------------
    # KPI Cards
    # ------------------------------------------------------

    st.markdown("## Key Performance Indicators")

    k1, k2, k3, k4 = st.columns(4)

    with k1:

        st.metric(
            "💰 Total Sales",
            f"${filtered['Sales'].sum():,.0f}"
        )

    with k2:

        st.metric(
            "🛒 Orders",
            f"{len(filtered):,}"
        )

    with k3:

        st.metric(
            "📦 Products",
            filtered["Product Name"].nunique()
        )

    with k4:

        st.metric(
            "🌍 Regions",
            filtered["Region"].nunique()
        )

    st.markdown("---")

    # ------------------------------------------------------
    # Sales by Year
    # ------------------------------------------------------

    st.subheader("📊 Total Sales by Year")

    yearly = (

        filtered

        .groupby("Year")["Sales"]

        .sum()

        .reset_index()

    )

    fig = px.bar(

        yearly,

        x="Year",

        y="Sales",

        color="Sales",

        text_auto=".2s",

        color_continuous_scale="Blues"

    )

    fig.update_layout(

        height=450,

        showlegend=False,

        xaxis_title="Year",

        yaxis_title="Sales"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    # ------------------------------------------------------
    # Monthly Trend
    # ------------------------------------------------------

    st.subheader("📈 Monthly Sales Trend")

    monthly = (

        filtered

        .groupby(

            pd.Grouper(

                key="Order Date",

                freq="ME"

            )

        )["Sales"]

        .sum()

        .reset_index()

    )

    fig = px.line(

        monthly,

        x="Order Date",

        y="Sales",

        markers=True

    )

    fig.update_layout(

        height=450,

        xaxis_title="Month",

        yaxis_title="Sales"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    # ------------------------------------------------------
    # Region vs Category
    # ------------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.subheader("Sales by Region")

        region_sales = (

            filtered

            .groupby("Region")["Sales"]

            .sum()

            .reset_index()

        )

        fig = px.bar(

            region_sales,

            x="Region",

            y="Sales",

            color="Region",

            text_auto=".2s"

        )

        fig.update_layout(

            showlegend=False,

            height=420

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    with right:

        st.subheader("Sales by Category")

        cat_sales = (

            filtered

            .groupby("Category")["Sales"]

            .sum()

            .reset_index()

        )

        fig = px.bar(

            cat_sales,

            x="Category",

            y="Sales",

            color="Category",

            text_auto=".2s"

        )

        fig.update_layout(

            showlegend=False,

            height=420

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    # ------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------

    st.subheader("🔥 Region vs Category Heatmap")

    heat = (

        filtered

        .pivot_table(

            index="Region",

            columns="Category",

            values="Sales",

            aggfunc="sum",

            fill_value=0

        )

    )

    fig = px.imshow(

        heat,

        text_auto=".2s",

        color_continuous_scale="Blues",

        aspect="auto"

    )

    fig.update_layout(

        height=550

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    # ------------------------------------------------------
    # Top Products
    # ------------------------------------------------------

    st.subheader("🏆 Top 10 Products")

    top_products = (

        filtered

        .groupby("Product Name")["Sales"]

        .sum()

        .sort_values(ascending=False)

        .head(10)

        .reset_index()

    )

    fig = px.bar(

        top_products,

        x="Sales",

        y="Product Name",

        orientation="h",

        color="Sales",

        text_auto=".2s",

        color_continuous_scale="Viridis"

    )

    fig.update_layout(

        height=600,

        yaxis_title="",

        xaxis_title="Sales"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    # ------------------------------------------------------
    # Raw Dataset
    # ------------------------------------------------------

    with st.expander("📄 View Filtered Dataset"):

        st.dataframe(

            filtered,

            use_container_width=True,

            hide_index=True

        )
    # ============================================================
# PAGE 2 : FORECAST EXPLORER
# ============================================================

elif page == "🔮 Forecast Explorer":

    st.title("🔮 Forecast Explorer")

    st.markdown(
        """
Generate future sales forecasts using **Prophet** and **XGBoost**.

Both models are trained automatically.

The model having the **lowest RMSE** is selected as the Best Model.
"""
    )

    st.markdown("---")

    # -------------------------------------------------------
    # FILTERS
    # -------------------------------------------------------

    left, right = st.columns(2)

    with left:

        forecast_level = st.selectbox(

            "Forecast Level",

            [

                "Overall",

                "Category",

                "Region"

            ]

        )

    with right:

        if forecast_level == "Category":

            selected_value = st.selectbox(

                "Select Category",

                sorted(df["Category"].unique())

            )

            filter_column = "Category"

        elif forecast_level == "Region":

            selected_value = st.selectbox(

                "Select Region",

                sorted(df["Region"].unique())

            )

            filter_column = "Region"

        else:

            selected_value = None

            filter_column = None

    horizon = st.select_slider(

        "Forecast Horizon (Months)",

        options=[1,2,3],

        value=3

    )

    st.markdown("---")

    # -------------------------------------------------------
    # PREPARE MONTHLY DATA
    # -------------------------------------------------------

    monthly = monthly_sales(

        df,

        filter_column,

        selected_value

    )

    if len(monthly) < 8:

        st.warning(

            "Not enough historical data for forecasting."

        )

        st.stop()

    # -------------------------------------------------------
    # GENERATE FORECASTS
    # -------------------------------------------------------

    with st.spinner("Training Prophet and XGBoost..."):

        prophet_result = prophet_forecast(

            monthly,

            periods=horizon

        )

        xgb_result = xgb_forecast(

            monthly,

            periods=horizon

        )

    available_models = []

    if prophet_result is not None:

        available_models.append(prophet_result)

    if xgb_result is not None:

        available_models.append(xgb_result)

    if len(available_models) == 0:

        st.error(

            "Unable to train forecasting models."

        )

        st.stop()

    best = min(

        available_models,

        key=lambda x: x["rmse"]

    )

    st.success(

        f"🏆 Best Model : {best['model']}"

    )

    # -------------------------------------------------------
    # METRICS
    # -------------------------------------------------------

    m1, m2, m3 = st.columns(3)

    with m1:

        st.metric(

            "Best Model",

            best["model"]

        )

    with m2:

        st.metric(

            "MAE",

            f"{best['mae']:,.2f}"

        )

    with m3:

        st.metric(

            "RMSE",

            f"{best['rmse']:,.2f}"

        )

    st.markdown("---")

    # -------------------------------------------------------
    # PREPARE FORECAST TABLE
    # -------------------------------------------------------

    future = best["forecast"].copy()

    future = future.rename(

        columns={

            "ds":"Date",

            "yhat":"Forecast"

        }

    )

    future["Forecast"] = future["Forecast"].round(2)

    future["Date"] = pd.to_datetime(

        future["Date"]

    )

    # -------------------------------------------------------
    # HISTORICAL TABLE
    # -------------------------------------------------------

    history = monthly.rename(

        columns={

            "ds":"Date",

            "y":"Sales"

        }

    )

    history["Type"] = "Historical"

    future["Type"] = "Forecast"

    forecast_df = pd.concat(

        [

            history,

            future.rename(

                columns={

                    "Forecast":"Sales"

                }

            )

        ],

        ignore_index=True

    )

    st.markdown("### Forecast Generated Successfully")
        # -------------------------------------------------------
    # FORECAST CHART
    # -------------------------------------------------------

    st.subheader("📈 Historical vs Forecast")

    fig = go.Figure()

    # Historical Sales
    fig.add_trace(
        go.Scatter(
            x=history["Date"],
            y=history["Sales"],
            mode="lines+markers",
            name="Historical Sales",
            line=dict(color="#1f77b4", width=3)
        )
    )

    # Forecast
    fig.add_trace(
        go.Scatter(
            x=future["Date"],
            y=future["Forecast"],
            mode="lines+markers",
            name=f"{best['model']} Forecast",
            line=dict(color="red", width=3, dash="dash"),
            marker=dict(size=9)
        )
    )

    fig.update_layout(
        height=550,
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Sales",
        legend=dict(
            orientation="h",
            y=1.05,
            x=1,
            xanchor="right"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -------------------------------------------------------
    # FORECAST TABLE
    # -------------------------------------------------------

    st.subheader("📅 Forecast Values")

    display_table = future.copy()

    display_table = display_table.rename(
        columns={
            "Date": "Forecast Month",
            "Forecast": "Forecast Sales"
        }
    )

    display_table["Forecast Month"] = (
        display_table["Forecast Month"]
        .dt.strftime("%B %Y")
    )
    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True
    )

    # -------------------------------------------------------
    # MODEL COMPARISON
    # -------------------------------------------------------

    st.subheader("📊 Prophet vs XGBoost Performance")

    comparison = []

    if prophet_result is not None:

        comparison.append({

            "Model":"Prophet",

            "MAE":round(prophet_result["mae"],2),

            "RMSE":round(prophet_result["rmse"],2)

        })

    if xgb_result is not None:

        comparison.append({

            "Model":"XGBoost",

            "MAE":round(xgb_result["mae"],2),

            "RMSE":round(xgb_result["rmse"],2)

        })

    comparison_df = pd.DataFrame(comparison)

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    # -------------------------------------------------------
    # COMPARISON BAR CHART
    # -------------------------------------------------------

    compare_plot = comparison_df.melt(

        id_vars="Model",

        var_name="Metric",

        value_name="Value"

    )

    fig = px.bar(

        compare_plot,

        x="Model",

        y="Value",

        color="Metric",

        barmode="group",

        text_auto=".2f"

    )

    fig.update_layout(

        height=450,

        yaxis_title="Error",

        xaxis_title="Model"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    # -------------------------------------------------------
    # DOWNLOAD FORECAST
    # -------------------------------------------------------

    st.subheader("⬇ Download Forecast")

    csv = future.to_csv(index=False).encode("utf-8")

    st.download_button(

        label="📥 Download Forecast CSV",

        data=csv,

        file_name="sales_forecast.csv",

        mime="text/csv"

    )

    # -------------------------------------------------------
    # MODEL SUMMARY
    # -------------------------------------------------------

    with st.expander("📘 Forecast Summary", expanded=True):

        st.write(f"**Best Model:** {best['model']}")

        st.write(f"**Forecast Horizon:** {horizon} Month(s)")

        st.write(f"**Mean Absolute Error (MAE):** {best['mae']:.2f}")

        st.write(f"**Root Mean Squared Error (RMSE):** {best['rmse']:.2f}")

        st.success(
            "The model with the lowest RMSE has been automatically selected as the best forecasting model."
        )

        # ============================================================
# PAGE 3 : ANOMALY REPORT
# ============================================================

elif page == "🚨 Anomaly Report":

    st.title("🚨 Sales Anomaly Report")

    st.markdown(
        """
Identify unusual spikes and drops in sales using
**Isolation Forest**.
        """
    )

    # ----------------------------------------------------------
    # CREATE WEEKLY SALES
    # ----------------------------------------------------------

    weekly = (

        df

        .groupby(

            pd.Grouper(

                key="Order Date",

                freq="W"

            )

        )["Sales"]

        .sum()

        .reset_index()

    )

    # ----------------------------------------------------------
    # TRAIN ISOLATION FOREST
    # ----------------------------------------------------------

    # Lazy import: see note at top of file.
    from sklearn.ensemble import IsolationForest

    model = IsolationForest(

        contamination=0.05,

        random_state=42,

        n_jobs=1

    )

    weekly["Anomaly"] = model.fit_predict(

        weekly[["Sales"]]

    )

    weekly["Label"] = weekly["Anomaly"].map(

        {

            1:"Normal",

            -1:"Anomaly"

        }

    )

    anomalies = weekly[

        weekly["Label"]=="Anomaly"

    ]

    # ----------------------------------------------------------
    # KPI CARDS
    # ----------------------------------------------------------

    c1,c2,c3 = st.columns(3)

    with c1:

        st.metric(

            "Weeks",

            len(weekly)

        )

    with c2:

        st.metric(

            "Anomalies",

            len(anomalies)

        )

    with c3:

        percentage = (

            len(anomalies)

            /

            len(weekly)

        )*100

        st.metric(

            "Anomaly Rate",

            f"{percentage:.1f}%"

        )

    st.markdown("---")

    # ----------------------------------------------------------
    # ANOMALY CHART
    # ----------------------------------------------------------

    st.subheader("Weekly Sales Trend")

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=weekly["Order Date"],

            y=weekly["Sales"],

            mode="lines",

            line=dict(

                color="#1f77b4",

                width=3

            ),

            name="Weekly Sales"

        )

    )

    fig.add_trace(

        go.Scatter(

            x=anomalies["Order Date"],

            y=anomalies["Sales"],

            mode="markers",

            marker=dict(

                color="red",

                size=11,

                symbol="circle"

            ),

            name="Anomaly"

        )

    )

    fig.update_layout(

        template="plotly_white",

        height=550,

        hovermode="x unified",

        xaxis_title="Week",

        yaxis_title="Sales"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    # ----------------------------------------------------------
    # ANOMALY TABLE
    # ----------------------------------------------------------

    st.subheader("Detected Anomalies")

    anomaly_table = anomalies.copy()

    anomaly_table = anomaly_table.rename(

        columns={

            "Order Date":"Date",

            "Sales":"Sales Value"

        }

    )

    anomaly_table["Date"] = anomaly_table["Date"].dt.strftime(

        "%d-%b-%Y"

    )

    anomaly_table["Sales Value"] = anomaly_table[

        "Sales Value"

    ].round(2)

    st.dataframe(

        anomaly_table[

            [

                "Date",

                "Sales Value"

            ]

        ],

        use_container_width=True,

        hide_index=True

    )

    # ----------------------------------------------------------
    # ANOMALY BAR CHART
    # ----------------------------------------------------------

    st.subheader("Anomaly Sales")

    fig = px.bar(

        anomaly_table,

        x="Date",

        y="Sales Value",

        color="Sales Value",

        text_auto=".2s",

        color_continuous_scale="Reds"

    )

    fig.update_layout(

        template="plotly_white",

        height=450,

        xaxis_title="Date",

        yaxis_title="Sales"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    # ----------------------------------------------------------
    # MONTHLY DISTRIBUTION
    # ----------------------------------------------------------

    st.subheader("Monthly Sales Distribution")

    monthly_sales_dist = (

        df

        .groupby(

            pd.Grouper(

                key="Order Date",

                freq="ME"

            )

        )["Sales"]

        .sum()

        .reset_index()

    )

    fig = px.box(

        monthly_sales_dist,

        y="Sales",

        points="all"

    )

    fig.update_layout(

        template="plotly_white",

        height=450

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    # ----------------------------------------------------------
    # DOWNLOAD
    # ----------------------------------------------------------

    csv = anomaly_table.to_csv(

        index=False

    ).encode(

        "utf-8"

    )

    st.download_button(

        "📥 Download Anomaly Report",

        data=csv,

        file_name="anomaly_report.csv",

        mime="text/csv"

    )

    # ----------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------

    with st.expander(

        "📘 Interpretation",

        expanded=True

    ):

        st.write(

            """
- 🔴 Red points represent unusual sales.

- Large spikes generally indicate promotional campaigns,
  seasonal demand, or festive sales.

- Sudden drops may indicate stock shortages,
  logistics delays, supplier issues,
  or unexpected market conditions.

- Isolation Forest automatically identifies
  these observations without requiring labels.
"""
        )

        # ============================================================
# PAGE 4 : PRODUCT DEMAND SEGMENTS
# ============================================================

elif page == "📦 Product Demand Segments":

    st.title("📦 Product Demand Segments")

    st.markdown("""
Product sub-categories are grouped into different demand segments using **K-Means Clustering**.

The visualization below shows the clusters projected into two dimensions using **PCA (Principal Component Analysis)**.
    """)

    # ----------------------------------------------------------
    # VALIDATION
    # ----------------------------------------------------------

    if segments_df.empty:

        st.error("Product_Demand_Segments.csv not found.")

        st.stop()

    # ----------------------------------------------------------
    # KPI CARDS
    # ----------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Sub Categories",
            segments_df["Sub-Category"].nunique()
        )

    with c2:

        st.metric(
            "Demand Segments",
            segments_df["Demand Segment"].nunique()
        )

    with c3:

        st.metric(
            "Total Sales",
            f"${segments_df['TotalSales'].sum():,.0f}"
        )

    with c4:

        st.metric(
            "Average Order Value",
            f"${segments_df['AverageOrderValue'].mean():.2f}"
        )

    st.markdown("---")

    # ----------------------------------------------------------
    # PCA CLUSTER CHART
    # ----------------------------------------------------------

    st.subheader("Demand Cluster Visualization")

    fig = px.scatter(

        segments_df,

        x="PC1",

        y="PC2",

        color="Demand Segment",

        text="Sub-Category",

        size="TotalSales",

        hover_data=[

            "GrowthRate",

            "Volatility",

            "AverageOrderValue"

        ],

        height=650

    )

    fig.update_traces(

        textposition="top center"

    )

    fig.update_layout(

        template="plotly_white",

        xaxis_title="Principal Component 1",

        yaxis_title="Principal Component 2"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    # ----------------------------------------------------------
    # SEGMENT DISTRIBUTION
    # ----------------------------------------------------------

    st.subheader("Products in each Demand Segment")

    count_df = (

        segments_df

        .groupby("Demand Segment")

        .size()

        .reset_index(name="Products")

    )

    fig = px.bar(

        count_df,

        x="Demand Segment",

        y="Products",

        color="Demand Segment",

        text_auto=True,

        height=450

    )

    fig.update_layout(

        template="plotly_white"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    # ----------------------------------------------------------
    # TOTAL SALES BY SEGMENT
    # ----------------------------------------------------------

    st.subheader("Total Sales by Demand Segment")

    sales_df = (

        segments_df

        .groupby("Demand Segment")["TotalSales"]

        .sum()

        .reset_index()

    )

    fig = px.bar(

        sales_df,

        x="Demand Segment",

        y="TotalSales",

        color="Demand Segment",

        text_auto=".2s",

        height=450

    )

    fig.update_layout(

        template="plotly_white",

        yaxis_title="Total Sales"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    # ----------------------------------------------------------
    # SEGMENT TABLE
    # ----------------------------------------------------------

    st.subheader("Sub-Categories by Demand Segment")

    table = segments_df.copy()

    table["GrowthRate"] = (

        table["GrowthRate"] * 100

    ).round(2)

    table["TotalSales"] = table["TotalSales"].round(2)

    table["AverageOrderValue"] = (

        table["AverageOrderValue"]

    ).round(2)

    table["Volatility"] = (

        table["Volatility"]

    ).round(2)

    st.dataframe(

        table[

            [

                "Sub-Category",

                "Demand Segment",

                "TotalSales",

                "GrowthRate",

                "Volatility",

                "AverageOrderValue"

            ]

        ],

        use_container_width=True,

        hide_index=True

    )

    # ----------------------------------------------------------
    # DOWNLOAD TABLE
    # ----------------------------------------------------------

    csv = table.to_csv(

        index=False

    ).encode("utf-8")

    st.download_button(

        "📥 Download Cluster Report",

        data=csv,

        file_name="product_demand_segments.csv",

        mime="text/csv"

    )

    # ----------------------------------------------------------
    # BUSINESS INSIGHTS
    # ----------------------------------------------------------

    st.markdown("---")

    st.subheader("Business Insights")

    top_segment = (

        sales_df

        .sort_values(

            "TotalSales",

            ascending=False

        )

        .iloc[0]["Demand Segment"]

    )

    st.success(

        f"🏆 Highest Revenue Segment : {top_segment}"

    )

    with st.expander(

        "Recommendation",

        expanded=True

    ):

        st.write("""

### Recommended Actions

• Increase inventory for High Demand products.

• Monitor Medium Demand products closely.

• Apply promotions to Low Demand products.

• High Volatility products require better forecasting.

• Products with high Average Order Value should receive premium marketing.

• Cluster analysis helps optimize inventory planning and warehouse utilization.

""")
        
