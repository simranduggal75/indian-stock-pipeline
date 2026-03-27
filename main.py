#Imports 
import yfinance as yf 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error 
from sentiment import analyze_sentiment 
import os 

# Indian stocks on NSE 
stocks = ["RELIANCE.NS", "TCS.NS" , "INFY.NS", "HDFCBANK.NS", "WIPRO.NS"]
DATA_PATH = "data/raw_stock_data.csv" 
print(DATA_PATH) 
os.makedirs("data", exist_ok=True) 
#fetch data 

if os.path.exists(DATA_PATH): 
    print("📂 Loading data from CSV...") 
    data = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True) 
else: 
    print("🌐 Fetching data from yfinance...") 
    data = yf.download(stocks, period="3mo", interval="1d")["Close"] 
    data.to_csv(DATA_PATH) 
    print(" Data fetched and saved!") 
    
def predict_stock(data, stock_name):

    print(f"\n🔍 Processing {stock_name}...")

    # Prepare data
    stock_data = data[[stock_name]].copy()
    stock_data = stock_data.dropna()
    stock_data.rename(columns={stock_name: "Close"}, inplace=True)

    # Feature Engineering
    stock_data["Days"] = range(len(stock_data))
    stock_data["Returns"] = stock_data["Close"].pct_change()
    stock_data["MA_5"] = stock_data["Close"].rolling(5).mean()
    stock_data["MA_10"] = stock_data["Close"].rolling(10).mean()

    stock_data = stock_data.dropna()

    # ML data
    X = stock_data[["Days", "MA_5", "MA_10", "Returns"]]
    y = stock_data["Close"]

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Evaluate
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    print(f"📊 Model MAE: {mae:.2f}")

    # Predict future
    last_day = stock_data["Days"].iloc[-1]
    last_ma5 = stock_data["MA_5"].iloc[-1]
    last_ma10 = stock_data["MA_10"].iloc[-1]
    last_return = stock_data["Returns"].iloc[-1]

    future_data = pd.DataFrame({
        "Days": [last_day + i for i in range(1, 8)],
        "MA_5": [last_ma5]*7,
        "MA_10": [last_ma10]*7,
        "Returns": [last_return]*7
    })

    predictions = model.predict(future_data)

    
    # NEWS SENTIMENT 
    
    news = [
        f"{stock_name} shows strong growth in recent quarter",
        f"{stock_name} faces market pressure and volatility"
    ]

    sentiments = analyze_sentiment(news)

    print("\n📰 News Sentiment Analysis:")
    for n, s in sentiments:
        print(f"{n} → {s}")

    
    # SAVE DATA 
    
    stock_data.to_csv(f"data/{stock_name}_processed.csv", index=False)

    pred_df = pd.DataFrame({
        "Day": range(1, 8),
        "Predicted Price": predictions
    })
    pred_df.to_csv(f"data/{stock_name}_predictions.csv", index=False)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(stock_data["Days"], stock_data["Close"], label="Actual Price")
    plt.plot(
        range(last_day + 1, last_day + 8),
        predictions,
        linestyle="--",
        label="Predicted Price"
    )

    plt.title(f"{stock_name} Price Prediction")
    plt.xlabel("Days")
    plt.ylabel("Price (INR ₹)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print predictions
    print(f"\n📈 {stock_name} Next 7 Days Prediction:")
    for i, price in enumerate(predictions, 1):
        print(f"Day {i}: ₹{price:.2f}")


# Call the prediction function for each stock
for stock_name in stocks:
    predict_stock(data, stock_name)
