import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sentiment import analyze_sentiment
import os

st.set_page_config(page_title="AI Stock Insight Engine", layout="wide")
st.title("📈 AI Stock Insight Engine")
st.caption("Live predictions for top Indian NSE stocks")

stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "WIPRO.NS"]

@st.cache_data
def load_data():
    os.makedirs("data", exist_ok=True)
    data = yf.download(stocks, period="3mo", interval="1d")["Close"]
    return data

def predict_stock(data, stock_name):
    stock_data = data[[stock_name]].copy().dropna()
    stock_data.rename(columns={stock_name: "Close"}, inplace=True)
    stock_data["Days"] = range(len(stock_data))
    stock_data["Returns"] = stock_data["Close"].pct_change()
    stock_data["MA_5"] = stock_data["Close"].rolling(5).mean()
    stock_data["MA_10"] = stock_data["Close"].rolling(10).mean()
    stock_data = stock_data.dropna()

    X = stock_data[["Days", "MA_5", "MA_10", "Returns"]]
    y = stock_data["Close"]
    model = LinearRegression()
    model.fit(X, y)
    mae = mean_absolute_error(y, model.predict(X))

    last_day = stock_data["Days"].iloc[-1]
    future_data = pd.DataFrame({
        "Days": [last_day + i for i in range(1, 8)],
        "MA_5": [stock_data["MA_5"].iloc[-1]] * 7,
        "MA_10": [stock_data["MA_10"].iloc[-1]] * 7,
        "Returns": [stock_data["Returns"].iloc[-1]] * 7
    })
    predictions = model.predict(future_data)

    news = [
        f"{stock_name} shows strong growth in recent quarter",
        f"{stock_name} faces market pressure and volatility"
    ]
    sentiments = analyze_sentiment(news)
    positive_count = sum(1 for _, s in sentiments if s == "Positive")
    negative_count = sum(1 for _, s in sentiments if s == "Negative")
    trend = "upward" if stock_data["MA_5"].iloc[-1] > stock_data["MA_10"].iloc[-1] else "downward"

    return stock_data, predictions, mae, sentiments, positive_count, negative_count, trend, last_day

with st.spinner("Loading stock data..."):
    data = load_data()

for stock_name in stocks:
    st.divider()
    st.subheader(f"🔍 {stock_name}")

    stock_data, predictions, mae, sentiments, pos, neg, trend, last_day = predict_stock(data, stock_name)

    # Chart — exactly like your local figures
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(stock_data["Days"], stock_data["Close"], label="Actual Price")
    ax.plot(
        range(last_day + 1, last_day + 8),
        predictions,
        linestyle="--",
        color="orange",
        label="Predicted Price"
    )
    ax.set_title(f"{stock_name} Price Prediction")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price (INR ₹)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    plt.close()

    # MAE
    st.markdown(f"**📊 Model MAE:** ₹{mae:.2f}")

    # Sentiment
    st.markdown("**📰 News Sentiment Analysis:**")
    for n, s in sentiments:
        st.write(f"{n} → **{s}**")

    # Insight
    st.markdown("**🤖 AI Insight:**")
    if pos > neg and trend == "upward":
        st.success("Stock shows a positive outlook due to strong sentiment and upward trend.")
    elif neg > pos and trend == "downward":
        st.error("Stock may decline due to negative sentiment and downward trend.")
    else:
        st.warning("Stock shows mixed signals. Further analysis recommended.")

    # Predictions — same as your terminal output
    st.markdown(f"**📈 {stock_name} Next 7 Days Prediction:**")
    for i, price in enumerate(predictions, 1):
        st.write(f"Day {i}: ₹{price:.2f}")