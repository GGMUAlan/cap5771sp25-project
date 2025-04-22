import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)

st.set_page_config(page_title="COVID-19 Recovery Rate and Testing Dashboard", layout="wide")
st.title("🦠 Global COVID-19 Recovery Dashboard")

MERGED_PATH = "/Users/yuexilyu/.cache/kagglehub/datasets/imdevskp/corona-virus-report/versions/166/df_final_merged_unscaled.csv"
CLEAN_PATH = "/Users/yuexilyu/.cache/kagglehub/datasets/imdevskp/corona-virus-report/versions/166/covid_19_clean_complete.csv"
MLP_PATH = "/Users/yuexilyu/jupter_deeplearning/cap5771sp25-project/Code/df_final_all.csv"

# Load data
df = pd.read_csv(MERGED_PATH)
df_clean = pd.read_csv(CLEAN_PATH)
df_mlp = pd.read_csv(MLP_PATH)

if 'recovery_rate_full' not in df_mlp.columns:
    if 'Recovered_full' in df_mlp.columns and 'Confirmed_full' in df_mlp.columns:
        df_mlp['recovery_rate_full'] = df_mlp['Recovered_full'] / df_mlp['Confirmed_full'].replace(0, np.nan)
    else:
        st.error("Cannot find necessary columns to compute recovery_rate_full.")
        st.stop()

# Convert date columns
df['Date'] = pd.to_datetime(df['Date'])
df_clean['Date'] = pd.to_datetime(df_clean['Date'])
df_mlp['Date'] = pd.to_datetime(df_mlp['Date'])

# Filter valid recovery rates
df = df[(df["recovery_rate"] >= 0) & (df["recovery_rate"] <= 1)]
df_latest = df.sort_values("Date").groupby("Country/Region").last().reset_index()

# KPI Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Average Recovery Rate", f"{df_latest['recovery_rate'].mean():.2%}")
col2.metric("Max Recovery Rate", f"{df_latest['recovery_rate'].max():.2%}")
col3.metric("Min Recovery Rate", f"{df_latest['recovery_rate'].min():.2%}")

# Maps
st.subheader("Global Recovery Rate Map")
st.plotly_chart(px.choropleth(df_latest, locations="Country/Region", locationmode="country names", color="recovery_rate", color_continuous_scale="Greens", range_color=[0, 1], labels={'recovery_rate': 'Recovery Rate'}, title="Recovery Rate by Country"), use_container_width=True)

st.subheader("Testing per Million Map")
st.plotly_chart(px.choropleth(df_latest, locations="Country/Region", locationmode="country names", color="Tests/1M pop", color_continuous_scale="Blues", labels={'Tests/1M pop': 'Tests per 1M'}, title="Testing per Million by Country"), use_container_width=True)

# Country Recovery Trend
st.subheader("Country Recovery Trend")
country_list = df_clean["Country/Region"].unique()
selected_country = st.selectbox("Select a country:", sorted(country_list))

country_df = df_clean[df_clean["Country/Region"] == selected_country].sort_values("Date")

if "Recovered" in country_df.columns and country_df["Recovered"].notna().sum() > 1:
    st.plotly_chart(px.line(country_df, x="Date", y="Recovered", title=f"{selected_country} Recovery Trend", labels={"Recovered": "Recovered", "Date": "Date"}), use_container_width=True)
else:
    st.warning("Not enough recovery data available.")

# Forecasting with Random Forest
st.subheader("Forecast Future Recoveries")
predict_days = st.slider("Days to predict (Random Forest):", min_value=3, max_value=30, value=7, key="rf_slider")
if country_df["Recovered"].notna().sum() > 1:
    df_model = country_df[["Date", "Recovered"]].dropna().copy()
    df_model["days_since"] = (df_model["Date"] - df_model["Date"].min()).dt.days
    X = df_model[["days_since"]].values
    y = df_model["Recovered"].values

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    threshold =  y
    accuracy = np.mean(np.abs(y - y_pred) <= threshold) * 100

    st.markdown(f"**Random Forest Evaluation:**")
    st.markdown(f"- Accuracy within range: `{accuracy:.2f}%`")

    future_dates = [df_model["Date"].max() + timedelta(days=i) for i in range(1, predict_days+1)]
    future_days = np.array([(d - df_model["Date"].min()).days for d in future_dates]).reshape(-1, 1)
    future_preds = model.predict(future_days)

    df_pred = pd.concat([df_model[["Date", "Recovered"]], pd.DataFrame({"Date": future_dates, "Recovered": future_preds})])

    fig_pred = px.line(df_pred, x="Date", y="Recovered", title=f"{selected_country} RF Recovery Forecast for {predict_days} Days", labels={"Recovered": "Recovered", "Date": "Date"})
    fig_pred.add_scatter(x=future_dates, y=future_preds, mode="lines", line=dict(dash="dash"), name="RF Forecast")
    st.plotly_chart(fig_pred, use_container_width=True)
else:
    st.warning("Not enough data for Random Forest forecasting.")

