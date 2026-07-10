# 📈 End-to-End Sales Forecasting & Demand Intelligence System

An AI-powered Sales Forecasting & Demand Intelligence System built using **Python**, **Streamlit**, **Prophet**, **XGBoost**, **SARIMA**, and **Machine Learning**. The project predicts future sales, detects anomalies, segments products by demand patterns, and provides an interactive business dashboard for inventory planning and decision-making.

---

## 🚀 Project Overview

Retail businesses must accurately predict product demand to avoid:

- Overstocking (higher storage costs)
- Understocking (lost sales and dissatisfied customers)

This project develops a complete demand intelligence pipeline that:

- Forecasts future sales using multiple forecasting models
- Detects unusual sales spikes and drops
- Segments products based on demand behavior
- Presents insights through an interactive Streamlit dashboard

---

## 🎯 Problem Statement

The objective is to build an intelligent forecasting system capable of:

- Predicting future product demand
- Comparing multiple forecasting models
- Detecting anomalous sales patterns
- Segmenting products into demand groups
- Assisting inventory and supply chain planning

---

# 📊 Dataset

**Primary Dataset**

- Sample Superstore Sales Dataset

**Supplementary Dataset**

- Public Holiday Calendar Dataset

---

# 🛠️ Technologies Used

- Python 3.11
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Plotly
- Scikit-learn
- XGBoost
- Statsmodels
- Prophet
- Joblib

---

# 📂 Project Structure

```text
SalesForecasting/
│
├── app.py
├── style.css
├── requirements.txt
├── runtime.txt
│
├── clean_sales_data.csv
├── prophet_model.pkl
├── xgb_model.pkl
├── kmeans_model.pkl
├── scaler.pkl
├── pca_model.pkl
├── cluster_df.pkl
│
├── notebooks/
│      analysis.ipynb
│
├── pages/
│      1_Sales_Dashboard.py
│      2_Forecast_Explorer.py
│      3_Anomaly_Report.py
│      4_Demand_Segmentation.py
│
├── plots/
│      Demand_Segments.png
│      Isolation_Forest_Anomalies.png
│      ZScore_Anomalies.png
│      Anomaly_Method_Comparison.png
│
└── README.md
```

---

# 📌 Features

### 📊 Sales Dashboard

- Total Sales KPI
- Monthly Sales Trend
- Year-wise Sales Analysis
- Region-wise Sales
- Category-wise Sales
- Interactive Filters
- Download Filtered Data

---

### 🔮 Forecast Explorer

- Prophet Forecast
- XGBoost Forecast
- 1–3 Month Forecast Horizon
- Interactive Forecast Charts
- MAE
- RMSE
- MAPE
- Forecast Table

---

### 🚨 Anomaly Detection

- Isolation Forest
- Z-Score Detection
- Weekly Sales Anomalies
- Business Explanation
- Anomaly Comparison

---

### 📦 Demand Segmentation

- K-Means Clustering
- Elbow Method
- PCA Visualization
- Product Demand Groups
- Stocking Recommendations

---

# 📈 Machine Learning Models

| Model | Purpose |
|---------|----------|
| SARIMA | Statistical Forecasting |
| Prophet | Time-Series Forecasting |
| XGBoost | Machine Learning Forecasting |
| Isolation Forest | Anomaly Detection |
| K-Means | Product Segmentation |
| PCA | Cluster Visualization |

---

# 📊 Evaluation Metrics

Models are evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

The best-performing model is selected based on these evaluation metrics.

---

# 📦 Product Demand Segments

Products are segmented into:

- High Volume, Stable Demand
- Growing Demand
- Low Volume, High Volatility
- Declining Demand

Each segment includes inventory recommendations for supply chain optimization.

---

# 📷 Dashboard Pages

🏠 Home

- Project Overview
- KPI Cards
- Dataset Summary

📊 Sales Dashboard

- Sales Analytics
- Filters
- Interactive Charts

🔮 Forecast Explorer

- Prophet Forecast
- XGBoost Forecast
- Performance Metrics

🚨 Anomaly Report

- Isolation Forest
- Z-Score
- Weekly Anomalies

📦 Demand Segmentation

- Cluster Visualization
- Product Groups
- Stocking Strategy

---

# ▶️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/SalesForecasting.git

cd SalesForecasting
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 🌐 Streamlit Deployment

The project can be deployed on **Streamlit Community Cloud**.

Required files:

- app.py
- requirements.txt
- runtime.txt

---

# 📈 Business Benefits

- Improved demand forecasting
- Better inventory management
- Reduced stock shortages
- Reduced overstock costs
- Early anomaly detection
- Data-driven supply chain decisions

---

# 🔮 Future Enhancements

- LSTM-based forecasting
- Deep Learning models
- Real-time API integration
- Inventory optimization
- Multi-store forecasting
- Cloud deployment
- Automated retraining pipeline

---

# 👨‍💻 Author

**Ankit Agrawal**

MCA (Artificial Intelligence)

Galgotias University

---

# ⭐ If you found this project useful, consider giving it a Star!
