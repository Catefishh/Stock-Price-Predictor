D# Stock Predictor Webapp using Dash

## Overview
This project implements a stock price predictor using a Long Short-Term Memory (LSTM) neural network. The predictor utilizes historical stock data fetched from Yahoo Finance to provide insights and forecasts for future stock prices. The project includes a web-based interface built with Dash to allow users to input stock tickers, view historical data, and generate predictions.

## Features
- **Interactive Dashboard**: Enter stock tickers and visualize historical stock prices.
- **Timeframe Selection**: Choose from predefined timeframes (`6 months`, `1 year`, `5 years`) for stock data.
- **Multi-step Forecasting**: Predict future stock prices for customizable horizons (e.g., 1 day, 1 week, etc.).
- **Data Table**: View historical and forecasted prices in a structured table.
- **Export to Excel**: Save historical data and predictions to an Excel file for further analysis.

## Project Structure
```
.
├── main.py               # Main script to run the Dash app and LSTM model integration
├── styles.css            # CSS for styling the web app
├── model.pth             # Pretrained LSTM model weights
├── requirements.txt      # List of dependencies for the project
```

## Installation
### Prerequisites
- Python 3.8 or above
- A virtual environment (optional but recommended)

### Steps
1. Download/Clone the files from the repository:

2. Create and activate a virtual environment (optional):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure the `model.pth` file is in the same directory as `main.py`.

## Usage
1. Run the application:
   ```bash
   python main.py
   ```
2. Open the app in your browser at `http://127.0.0.1:8050/`.
3. Interact with the interface:
   - Enter a stock ticker symbol (e.g., `AAPL` for Apple).
   - Select a timeframe for historical data.
   - Generate forecasts for future stock prices.
   - Export data to an Excel file.

## Files and Their Roles
- **`main.py`**: Contains the LSTM model class, Dash app setup, and all necessary callbacks for interactivity.
- **`styles.css`**: Adds styling to the web interface, enhancing usability and aesthetics.
- **`model.pth`**: Pretrained LSTM model for making stock price predictions.
- **`requirements.txt`**: Lists all Python libraries required to run the project.

## Model Details
The LSTM model is implemented in PyTorch with the following specifications:
- **Input Size**: 5 (Open, High, Low, Close, Volume)
- **Hidden Size**: 100
- **Layers**: 3
- **Dropout**: 20%
- **Output Size**: 1 (Predicted Close Price)

The model was trained on a subset of historical stock data from Yahoo Finance using Google Colab. 
Here is the link if you would like to train the model itself: https://colab.research.google.com/drive/1hTprbC44SkEvvzkxeR1VeBsHaTE7sBlR?usp=sharing

## Dependencies
- Dash
- Plotly
- Pandas
- yFinance
- PyTorch

All dependencies can be installed using `pip install -r requirements.txt`.

## Acknowledgments
- Yahoo Finance for stock data.
- Dash for creating an interactive web interface.
- PyTorch for building and training the LSTM model used in this project.

## License
This project is licensed under the MIT License.
