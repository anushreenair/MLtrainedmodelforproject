# StockWizard - Stock Price Prediction using LSTM

## Overview
StockWizard is a stock price prediction tool that utilizes a Long Short-Term Memory (LSTM) neural network to forecast stock prices based on historical data. The model is trained using Yahoo Finance data and visualizes both actual and predicted stock prices.

## Features
- Fetches historical stock price data from Yahoo Finance.
- Preprocesses data using MinMaxScaler.
- Builds an LSTM-based deep learning model.
- Trains the model on historical stock data.
- Predicts future stock prices based on the trained model.
- Visualizes actual vs. predicted stock prices.
- Saves the trained model for further use.

## Installation
Ensure you have Python installed, then install the required dependencies using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow yfinance
```

## Usage
1. Run the script:
   ```bash
   python stockwizard_trained_model.py
   ```
2. Enter a stock symbol when prompted (e.g., AAPL, GOOG, TSLA).
3. The script will fetch data, train the model, and display the predictions.
4. The trained model is saved as `stock_model.keras`.

## Model Details
- Uses a sequential LSTM network with:
  - Two LSTM layers (50 units each)
  - Dropout layers (20%)
  - Fully connected Dense layers
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam
- Training batch size: 16
- Epochs: 30

## Output
- Plots the historical closing prices.
- Displays actual vs. predicted prices.
- Saves the trained model for later use.

## Dependencies
- Python 3.x
- TensorFlow
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Yahoo Finance API (yfinance)

## Notes
- Ensure a stable internet connection for data fetching.
- The model's performance depends on the availability and quality of historical data.
- The prediction accuracy varies based on market volatility and trends.

## License
This project is licensed under the MIT License.

## Contributors
- Anushree

For any queries, feel free to contact me!

/*****************************************************************************************************************************************/


