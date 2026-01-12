# 🚗📈 Tesla Stock Price Prediction using Deep Learning
SimpleRNN & LSTM | Streamlit Dashboard | Financial Time-Series Forecasting

A production-grade deep learning project focused on forecasting Tesla (TSLA) stock prices using time-series neural networks, complete with model evaluation, comparison, and an interactive Streamlit dashboard.

---
---

## 🚀 Key Highlights

- End-to-end deep learning pipeline (EDA → Modeling → Deployment)
- Time-series forecasting using SimpleRNN & LSTM
- Multi-horizon prediction (1 to 30 trading days)
- Hyperparameter tuning using KerasTuner
- Objective model comparison & best-model selection
- Interactive Streamlit dashboard with professional UI
- Candlestick, trend, volatility & risk visualizations
- Buy / Sell / Hold signal interpretation
- Risk-adjusted performance using Sharpe Ratio
- Production-ready, modular, and extensible design

---
---

## 📌 Project Overview

Stock market data is sequential and temporal, making it well-suited for Recurrent Neural Networks (RNNs).

This project builds and compares SimpleRNN and LSTM models to predict Tesla’s closing stock price, evaluates their performance across multiple forecast horizons, and deploys the best model through a modern Streamlit dashboard.

The application is designed to simulate a real-world financial analytics system, focusing on:

- Forecast accuracy
- Stability across horizons
- Risk awareness
- Clear business interpretation


---
---

## 🎯 Problem Statement

Create a deep learning–based predictive system to forecast Tesla’s stock closing price with the following objectives:

- Model sequential price behavior using RNN-based architectures
- Predict future prices from 1 to 30 trading days
- Emphasize short-term decision horizons (1D, 5D, 10D)
- Compare baseline vs tuned models
- Select the best model using objective evaluation metrics
- Deploy the final solution as an interactive dashboard


---
---

## 📁 Project Structure
```
Tesla_Stock_Price_Predication/
│
├── assets/
│   └── styles.css
│
├── data/
│   └── TSLA.csv
│
├── models/
│   ├── lstm_hyperparameter_tuning/
│   │
│   ├── rnn_hyperparameter_tuning/
│   │
│   ├── best_forecasting_model.keras
│   │
│   ├── lstm_baseline_best.h5
│   ├── lstm_tuned_best.h5
│   ├── simple_rnn_baseline_best.h5
│   ├── simple_rnn_tuned_best.h5
│   │
│   └── scaler.pkl
│
├── notebooks/
│   └── Tesla_Stock_Price_Prediction.ipynb
│
├── reports/
│   ├── best_model_report.json
│   │
│   ├── best_model_summary.txt
│   │
│   └── model_metrics_summary.csv
│
├── venv/
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
└── .gitignore

```
---
---

## ⚙️ Installation & Setup (Step-by-Step)

### 1️⃣ Clone the Repository

```
git clone https://github.com/your-username/Tesla_Stock_Price_Prediction.git
```
```
cd Tesla_Stock_Price_Prediction
```


### 2️⃣ Create a Virtual Environment

You can use any of the following commands depending on your system:

#### Windows

```
py -m venv venv
```


#### macOS / Linux

```
python3 -m venv venv
```


#### Alternative (cross-platform)

```
python -m venv venv
```

### 3️⃣ Activate the Virtual Environment

#### Windows

```
venv\Scripts\activate
```

#### macOS / Linux

```
source venv/bin/activate
```

#### 4️⃣ Install Project Dependencies

```
pip install -r requirements.txt
```

### ▶️ Run the Streamlit Application

Once models and scaler are available:

```
streamlit run app.py
```

The dashboard will automatically load the best-performing model and display forecasts, analytics, and visual insights.

---
---

## 🧪 Model Training (Notebook Execution)

Before running the Streamlit application, the deep learning models must be trained.

📓 Training Notebook

Open the notebook: ```notebooks/Tesla_Stock_Prediction.ipynb```


Run all cells sequentially to perform:

- Exploratory Data Analysis (EDA)
- Data preprocessing & scaling
- Training baseline SimpleRNN and LSTM models
- Hyperparameter tuning using KerasTuner
- Model evaluation across multiple horizons

Automatic saving of:

- Best-performing model (.h5 / .keras)
- Scaler (.pkl)
- Model evaluation report (.json)

These artifacts are required for the Streamlit dashboard.

---
---

## ☁️ Recommended Option — Run in Google Colab

👉 The easiest way to generate all model files is to run the notebook in Google Colab.

Steps:

- Upload the notebook to Colab
- Upload TSLA.csv dataset
- Run all cells
- Download generated model artifacts
- Place them in the correct local folders


---
---

## 📦 Alternative — Download Complete Project (With Models)

You can skip the entire training process by downloading the fully prepared project:

🔗 Google Drive (Includes trained models & scaler)

link here

---
---

## ▶️ Before Running:

- Ensure dataset paths are correct

If using Colab, confirm Google Drive is mounted

📌 Note:
If you downloaded the project from Google Drive, you do NOT need to run the notebook unless you want to:

- Retrain models
- Modify datasets
- Experiment with architectures or hyperparameters

---
---

## 🧠 Deep Learning Models Used

### 1️⃣ SimpleRNN

- Captures short-term temporal dependencies
- Faster training
- Used as a baseline comparison model

### 2️⃣ LSTM (Long Short-Term Memory)

- Handles long-term dependencies
- Mitigates vanishing gradient issues
- Better suited for volatile financial time-series
- Shows superior generalization after tuning

### Each model was trained in:

- Baseline configuration
- Hyperparameter-tuned configuration


---
---

## 📊 Dataset Details

Dataset: Tesla Stock Price Data (TSLA)

Granularity: Daily prices

Features Available:

- Date
- Open
- High
- Low
- Close
- Adj Close
- Volume

Target Variable: Close price

Scaling: MinMaxScaler

Lookback Window: 60 trading days

Only the closing price was used for modeling, in line with the problem requirements.

Missing values (if any) were handled carefully to preserve time-series continuity, avoiding forward-looking bias.

---
---

## 🧪 Model Evaluation Strategy

### Metrics Used

| Metric | Purpose |
|------|--------|
| MSE | Penalizes large errors |
| RMSE | Measures absolute error magnitude |
| MAE | Average deviation |
| MAPE | Business-friendly percentage error |

**Primary Metric:** MAPE  
**Secondary Metrics:** MAE, RMSE  
**Stability Check:** Consistency across forecast horizons

---

### Forecast Horizons Evaluated

- **1-Day**
- **5-Day**
- **10-Day**

> Although the system supports forecasts up to **30 days**, formal evaluation focuses on **short-term horizons**, where deep learning models are empirically more reliable for financial data.

---
---

## 🏆 Final Model Selection

After comprehensive evaluation, the **Hyper-Tuned LSTM model** was selected as the final production model due to:

- **Lowest forecasting error** across key evaluation horizons  
- **Superior handling of long-term temporal dependencies**  
- **Stronger generalization performance** compared to SimpleRNN  
- **More stable and smoother forecast trajectories**  
- **Improved business interpretability** for decision support  

The selected model is **automatically loaded by default** in the Streamlit dashboard to ensure optimal forecasting performance.


---
---

## 🖥️ Streamlit Dashboard Features

### 🔹 Market Snapshot
- **Last Closing Price** of Tesla stock  
- **Market Bias** indicator (Bullish / Bearish) based on forecast direction  
- **Model Confidence Score**, adjusted for recent market volatility  

### 🔹 Forecast Outlook Cards
- **1-Day, 5-Day, and 10-Day** forecast outlook  
- Displays **absolute price change** and **percentage change**  
- Clear **green (upside)** and **red (downside)** visual cues for quick interpretation  

### 🔹 Visualizations
- **Candlestick chart** showing the last 60 trading days with forecast overlay  
- **Line trend chart** with highlighted forecast zone  
- **Forecast candlestick projection** to visualize expected price behavior  
- **Daily returns bar chart** for short-term price movement analysis  
- **Rolling volatility chart** to assess market risk  
- **Multi-model forecast comparison** (optional) to compare RNN vs LSTM behavior  

### 🔹 Advanced Analytics
- **Buy / Sell / Hold** trading signal based on forecasted returns  
- **Sharpe Ratio** to evaluate risk-adjusted performance  
- **Forecast confidence interpretation** for decision support  
- **Adjustable forecast horizon (1–30 days)** for flexible short-term analysis  


---
---

## 💼 Business Use Cases

- 📈 **Short-term trading & market trend analysis**  
  Identify near-term price direction and momentum using deep learning forecasts.

- 📊 **Risk-aware investment decision support**  
  Combine forecasted returns, volatility, and Sharpe Ratio for informed decisions.

- 🧪 **Model comparison for financial forecasting research**  
  Analyze performance differences between SimpleRNN and LSTM architectures.

- 🎓 **Educational demonstration of time-series deep learning**  
  Practical example of applying RNN-based models to real-world financial data.

- 🏢 **Portfolio-grade financial analytics dashboard**  
  A deployable, interactive dashboard suitable for portfolio and case-study use.

---
---

## 🚀 Future Enhancements

- 📰 **News sentiment analysis** using NLP for market context enrichment  
- 🔁 **Transformer-based time-series models** for long-range dependencies  
- 📊 **Probabilistic forecasting** with confidence intervals and uncertainty bands  
- 🌐 **Live market data API integration** for real-time predictions  
- 📈 **Multi-stock comparison dashboard** for cross-asset analysis  
- 🔄 **Automated retraining pipeline** for continuous model updates  

---
---

## ⚠️ Disclaimer

This project is intended for **educational and analytical purposes only**.  
It does **not** constitute financial, investment, or trading advice.  

Stock markets are inherently volatile and unpredictable.  
Predictions generated by this system should be interpreted as **directional insights**, not guaranteed outcomes.


---
---
## 🤝 Author

### **Predeep Kumar**

🧑‍💻 **AI Engineer | Deep Learning | Time-Series Forecasting**

Built with ❤️ as a **production-ready deep learning financial forecasting project**,  
demonstrating strong model design, rigorous evaluation, and deployment-ready analytics.


