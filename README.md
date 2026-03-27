# 📈 Ai-stock-insight-engine

## 🚀 Overview
This project is an end-to-end stock analysis pipeline that combines **Machine Learning, Feature Engineering, and AI-based Sentiment Analysis** to predict stock prices and generate intelligent insights.

It fetches real-time stock data, processes it, predicts future prices, analyzes news sentiment, and provides **interpretable AI-driven insights**.


## 🧠 Key Features

- 📊 **Stock Data Pipeline**
  - Fetches stock data using Yahoo Finance API
  - Caches data locally for efficiency

- ⚙️ **Feature Engineering**
  - Moving Averages (MA_5, MA_10)
  - Daily Returns
  - Time-based features

- 🤖 **Machine Learning Model**
  - Linear Regression for price prediction
  - Predicts next 7 days stock prices
  - Model evaluation using MAE (Mean Absolute Error)

- 📰 **Sentiment Analysis**
  - Uses TextBlob to analyze financial news sentiment
  - Classifies news as Positive / Negative / Neutral

- 🧩 **AI Insights (Decision Layer)**
  - Combines sentiment + trend indicators
  - Generates human-readable insights:
     "Stock shows positive outlook due to strong sentiment and upward trend"


## 📂 Project Structure


ai-stock-insight-engine/
│
├── main.py                 # Main pipeline script
├── sentiment.py            # Sentiment analysis module
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
├── data/                   # Generated CSV files (ignored)
└── analysis.py             #  experimentation


## ⚙️ Installation & Setup

### 1. Clone the repository

git clone https://github.com/simranduggal75/ai-stock-insight-engine.git
cd ai-stock-insight-engine


### 2. Install dependencies

pip install -r requirements.txt

## ▶️ Run the Project

python main.py


## 📊 Sample Output


🔍 Processing RELIANCE.NS...
📊 Model MAE: 45.23

📰 News Sentiment Analysis:
Reliance shows strong growth → Positive
Reliance faces pressure → Negative

🤖 AI Insight:
Stock shows mixed signals. Further analysis recommended.

📈 Next 7 Days Prediction:
Day 1: ₹2450.23
Day 2: ₹2462.11

## 📈 How It Works

1. Fetch stock data (yfinance)
2. Apply feature engineering (MA, returns)
3. Train regression model
4. Predict future prices
5. Analyze sentiment from news
6. Combine signals → generate AI insight



## 🚀 Future Improvements

* Integrate real-time news APIs
* Use advanced models (LSTM, XGBoost)
* Add Streamlit dashboard
* Deploy as web app


## 👩‍💻 Author

Simran Duggal
AI/ML Engineer 


## ⭐ If you found this useful

Give this repo a ⭐ on GitHub!

## 📸 Output Screenshot
  Please refer to screenshots folder for output screenshots

