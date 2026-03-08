# Stock Price Prediction and Portfolio Optimization

## Project Overview
This project focuses on predicting stock prices and designing efficient investment portfolios using statistical models, machine learning algorithms, and deep learning techniques.

The study analyzes stocks from multiple sectors of the Indian stock market and compares different predictive modeling approaches to determine the most effective methods for stock price forecasting and portfolio optimization.

The objective is to help investors make informed investment decisions by forecasting stock prices and constructing optimal portfolios that balance risk and return.

This project was developed as part of the Post Graduate Program in Data Science at Praxis Business School.

---

## Problem Statement

Stock markets are highly volatile and predicting stock price movements is a challenging task. Investors need reliable models that can analyze historical data and forecast future price trends.

This project aims to:

- Predict stock prices using statistical, econometric, machine learning, and deep learning models
- Compare model performance across multiple sectors
- Construct optimized portfolios using Modern Portfolio Theory
- Evaluate portfolio performance through backtesting

---

## Dataset

The dataset consists of historical stock price data from **January 2016 to August 2021** collected using the Yahoo Finance API.

Each dataset contains:

- Open price
- High price
- Low price
- Close price
- Trading volume
- Date

Additional engineered features include:

- Day of week
- Day of month
- Month
- Price range
- NIFTY50 index to capture overall market sentiment

---

## Sectors and Stocks Analyzed

Five major sectors of the Indian economy were selected:

- Metal
- Pharma
- IT
- Banking
- Automobile

Example stocks analyzed include:

- Tata Steel
- Sun Pharma
- Infosys
- HDFC Bank
- Maruti Suzuki

These stocks were selected based on their contribution to sectoral indices.

---

## Models Implemented

### Statistical & Econometric Models

- Multivariate Linear Regression
- ARIMA (Autoregressive Integrated Moving Average)
- VAR (Vector Autoregression)
- MARS (Multivariate Adaptive Regression Splines)

### Machine Learning Models

- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- XGBoost

### Deep Learning Models

- LSTM (Long Short-Term Memory)
- CNN (Convolutional Neural Network)

---

## Validation Strategy

The project uses **Walk-Forward Validation**, which is more suitable for time-series forecasting compared to traditional train-test split.

Two approaches were used:

- Expanding Window Validation
- Sliding Window Validation

These methods allow models to be trained on recent data, which is crucial for stock price prediction.

---

## Portfolio Optimization

Portfolio optimization was performed using:

- Minimum Variance Portfolio
- Optimal Risk Portfolio
- Equal Weight Portfolio

Historical stock prices were used for training and the portfolio performance was evaluated using **backtesting on future market data**.

---

## Performance Metric

Model performance was evaluated using:

RMSE / Mean Percentage

This metric helps compare prediction accuracy across stocks with different price ranges.

---


---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Statsmodels
- TensorFlow / Keras
- Matplotlib
- Seaborn
- Yahoo Finance API

---

## Key Insights

- Machine learning and deep learning models outperform traditional statistical models for stock price prediction.
- Walk-forward validation improves prediction reliability for time-series data.
- Portfolio optimization techniques help balance risk and return effectively.

---

## Future Improvements

- Incorporating sentiment analysis from financial news and social media
- Using transformer-based deep learning models
- Real-time stock prediction system
- Integration with live trading APIs

---

## Contributors

- Koushik Tulasi
- Ashwin Kumar R S
- Geetha Joseph
- Kaushik Muthukrishnan
- Praveen Varukolu

Supervisor: **Prof. Jaydip Sen**  
Praxis Business School

---

## License

This project is for educational and research purposes.
