import os
import io
import dash
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
import plotly.graph_objs as go
from dash import dcc, html, Input, Output, State, dash_table
from sklearn.preprocessing import MinMaxScaler

# Step 1: Reinstate the model class (LSTM). 
class LSTMModel(nn.Module):
    """
    LSTM model for time-series forecasting.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers.
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout_prob,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input feature tensor of shape (batch_size, sequence_length, input_size).
        Returns:
            torch.Tensor: Final output of shape (batch_size, output_size).
        """
        # Initialize hidden and cell states for LSTM.
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

        # Feeding data into the LSTM model.
        out, _ = self.lstm(x, (h0, c0))

        # Output is taken from the last time step.
        out = self.fc(out[:, -1, :])
        return out

# Step 2: Define model parameters and load the trained weights.
input_size = 5
hidden_size = 100
num_layers = 3
output_size = 1
dropout_prob = 0.2

model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_prob)

# Determine the current directory and load model weights.
current_dir = os.path.dirname(os.path.abspath(__file__))
model_filename = "model.pth"
model_path = os.path.join(current_dir, model_filename)

# Load the model's state dictionary.
model.load_state_dict(
    torch.load(
        model_path,
        map_location=torch.device('cpu'),
        weights_only=True
    )
)
model.eval()  # Set the model to evaluation mode.

# Step 3: Set up the Dash web app.
app = dash.Dash(__name__)

# Define available timeframes for fetching stock data.
TIMEFRAMES = {
    '6mo': {'period': '6mo', 'interval': '1d'},
    '1y': {'period': '1y', 'interval': '1d'},
    '5y': {'period': '5y', 'interval': '1d'},
}

# Step 4: Define the helper function to fetch data using yfinance.
def fetch_data(ticker, timeframe):
    """
    Fetch stock data from yfinance based on the selected timeframe.

    Args:
        ticker (str): Stock ticker symbol.
        timeframe (str): Key to TIMEFRAMES dict.

    Returns:
        pd.DataFrame: Historical stock data.
    """
    if not ticker:
        return pd.DataFrame()

    params = TIMEFRAMES.get(timeframe, TIMEFRAMES['1y'])
    data = yf.Ticker(ticker).history(period=params['period'], interval=params['interval'])
    return data

# Step 5: Define the layout of the Dash app.
app.layout = html.Div([
    html.H1("Stock Predictor"),
    html.Div([
        html.Label("Enter Stock Ticker:"),
        dcc.Input(id='ticker-input', type='text'),
    ]),
    html.Div([
        html.Label("Select timeframe of stock data to fetch:"),
        dcc.Dropdown(
            id='timeframe-dropdown',
            options=[{'label': k, 'value': k} for k in TIMEFRAMES.keys()],
            value='6mo'
        )
    ]),
    html.Button("Get Data", id='get-data-button', n_clicks=0),
    dcc.Loading([
        dcc.Graph(id='price-graph'),
        dash_table.DataTable(
            id='price-table',
            page_size=10,
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold',
                'fontSize': '16px'
            },
            style_cell={
                'textAlign': 'left',
                'padding': '5px'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ],
        )
    ]),
    html.Br(), html.Hr(),
    html.Label("Select Forecast Horizon:"),
    dcc.Dropdown(
        id='forecast-horizon-dropdown',
        options=[
            {'label': '1 Day', 'value': 1},
            {'label': '1 Week', 'value': 5},
            {'label': '1 Month', 'value': 22},
            {'label': '1 Year', 'value': 252},
        ],
        value=1
    ),
    html.Button("Forecast Future", id='forecast-button', n_clicks=0),
    html.Div(id='future-predictions-output'),
    html.Br(), html.Hr(),
    html.Button("Save to Excel", id='save-button'),
    dcc.Download(id="download-dataframe-xlsx")
])

# Step 6: Defining the multi-step forecasting function.
def multi_step_forecast(model, last_sequence, scaler, future_steps):
    """
    Generate multi-step forecast using the downloaded LSTM model from Colab.
    https://colab.research.google.com/drive/1hTprbC44SkEvvzkxeR1VeBsHaTE7sBlR?usp=sharing

    Args:
        model (nn.Module): The trained LSTM model for forecasting.
        last_sequence (torch.Tensor): The last sequence of scaled features.
        scaler (MinMaxScaler): Scaler used for feature normalization.
        future_steps (int): Number of future prediction steps.

    Returns:
        list[float]: List of scaled predictions for each future step.
    """
    model.eval()
    predictions = []

    # Copy the last sequence to start forecasting.
    current_seq = last_sequence.clone()

    with torch.no_grad():
        for _ in range(future_steps):
            # Get prediction for current sequence.
            pred = model(current_seq).item()  
            predictions.append(pred)

            # Prepare the next input sequence by shifting and adding the new prediction.
            next_features = current_seq[0, -1, :].clone()
            next_features[3] = pred  # Update the 'Close' price in features.
            next_seq = torch.cat(
                (current_seq[:, 1:, :], next_features.unsqueeze(0).unsqueeze(0)),
                dim=1
            )
            current_seq = next_seq

    return predictions

# Step 7: Dash callback to update graph and table when fetching data.
@app.callback(
    Output('price-graph', 'figure'),
    Output('price-table', 'data'),
    Output('price-table', 'columns'),
    Input('get-data-button', 'n_clicks'),
    State('ticker-input', 'value'),
    State('timeframe-dropdown', 'value')
)
def update_data(n_clicks, ticker, timeframe):
    """
    Fetches data from yfinance and updates the price graph and data table.

    Args:
        n_clicks (int): Number of times the "Get Data" button has been clicked.
        ticker (str): Stock ticker symbol from user input.
        timeframe (str): Selected timeframe from the dropdown.

    Returns:
        tuple: (figure, table_data, columns) for updating the Dash components.
    """
    if n_clicks == 0 or not ticker:
        return {}, [], []

    data = fetch_data(ticker, timeframe)
    if data.empty:
        return {}, [], []

    data.reset_index(inplace=True)
    # Rounded to two decimals for better readability.
    data['Open'] = data['Open'].round(2)
    data['Close'] = data['Close'].round(2)

    # Creating the figure with historical Open and Close prices.
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], mode='lines', name='Open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
    fig.update_layout(
        title=f"{ticker} Prices (No predictions)",
        xaxis_title="Date",
        yaxis_title="Price"
    )

    # Creating the table data and columns.
    table_data = data[['Date', 'Open', 'Close']].to_dict('records')
    columns = [
        {"name": "Date", "id": "Date"},
        {"name": "Open Price", "id": "Open", "type": "numeric", "format": {"specifier": ".2f"}},
        {"name": "Close Price", "id": "Close", "type": "numeric", "format": {"specifier": ".2f"}},
    ]

    return fig, table_data, columns

# Step 8: Dash callback to generate future forecasts.
@app.callback(
    Output('future-predictions-output', 'children'),
    Input('forecast-button', 'n_clicks'),
    State('ticker-input', 'value'),
    State('timeframe-dropdown', 'value'),
    State('forecast-horizon-dropdown', 'value')
)
def forecast_future(n_clicks, ticker, timeframe, future_steps):
    """
    Generates future stock price predictions and displays them.

    Args:
        n_clicks (int): Number of times the "Forecast Future" button is clicked.
        ticker (str): Stock ticker symbol from user input.
        timeframe (str): Selected timeframe from the dropdown.
        future_steps (int): Number of future days to forecast.

    Returns:
        dash.html.Div: Contains a graph of historical + predicted data and a data table.
    """
    if n_clicks == 0 or not ticker:
        return "No forecast yet."

    # Fetch historical data for the given ticker.
    data = fetch_data(ticker, timeframe)
    if data.empty:
        return "No data found."

    data.reset_index(inplace=True)
    sequence_length = 60  
    if len(data) < sequence_length:
        return "Not enough data for forecasting."

    # Scale features for model input.
    features = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # Prepare the last sequence from historical data.
    last_seq = features_scaled[-sequence_length:]
    last_seq = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0)

    # Forecast future steps using the multi-step forecasting function.
    future_predictions_scaled = multi_step_forecast(model, last_seq, scaler, future_steps)

    # Inverse transform predictions to original scale.
    predicted_close_values = scaler.inverse_transform(
        [[0, 0, 0, pred, 0] for pred in future_predictions_scaled]
    )[:, 3]
    predicted_close_values = np.round(predicted_close_values, 2)

    # Generate future dates, skipping weekends.
    last_date = data['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps, freq='B')

    # Create DataFrame for future predictions.
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Close': predicted_close_values
    })

    # Step 9: Combining the historical and predicted data for visualization.
    combined_fig = go.Figure()
    combined_fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], mode='lines', name='Historical Open'))
    combined_fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Historical Close'))
    combined_fig.add_trace(go.Scatter(
        x=future_df['Date'], y=future_df['Predicted Close'],
        mode='lines+markers', name='Predicted Close'
    ))
    combined_fig.update_layout(
        title=f"Historical & Future Predictions for {ticker}",
        xaxis_title="Date",
        yaxis_title="Price"
    )

    # Step 10: Creating a table with the combined data (historical + predicted).
    historical_df = data[['Date', 'Open', 'Close']].copy()
    historical_df['Open'] = historical_df['Open'].round(2)
    historical_df['Close'] = historical_df['Close'].round(2)
    historical_df['Predicted Close'] = '' 

    future_df['Open'] = ''
    future_df['Close'] = ''
    future_df['Predicted Close'] = future_df['Predicted Close'].round(2)
    future_df = future_df[['Date', 'Open', 'Close', 'Predicted Close']]

    # Combine historical and future data into one DataFrame.
    combined_df = pd.concat([historical_df, future_df], ignore_index=True)

    # Convert combined data into a table-friendly format.
    table_data = combined_df.to_dict('records')
    columns = [
        {"name": "Date", "id": "Date"},
        {"name": "Open Price", "id": "Open", "type": "numeric", "format": {"specifier": ".2f"}},
        {"name": "Close Price", "id": "Close", "type": "numeric", "format": {"specifier": ".2f"}},
        {"name": "Predicted Close", "id": "Predicted Close", "type": "numeric", "format": {"specifier": ".2f"}},
    ]

    # Return the updated layout with graph and data table.
    return html.Div([
        dcc.Graph(figure=combined_fig),
        dash_table.DataTable(
            data=table_data,
            columns=columns,
            page_size=10,
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold',
                'fontSize': '16px'
            },
            style_cell={
                'textAlign': 'left',
                'padding': '5px'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ],
        )
    ])

# Step 11: Dash callback to save data and predictions to an Excel file.
@app.callback(
    Output("download-dataframe-xlsx", "data"),
    Input("save-button", "n_clicks"),
    State('ticker-input', 'value'),
    State('timeframe-dropdown', 'value'),
    prevent_initial_call=True,
)
def save_to_excel(n_clicks, ticker, timeframe):
    """
    Generates an Excel file with historical data and model predictions.

    Args:
        n_clicks (int): Number of times the "Save to Excel" button is clicked.
        ticker (str): Stock ticker symbol.
        timeframe (str): Selected timeframe key from TIMEFRAMES.

    Returns:
        dcc.send_bytes: The Excel file for download.
    """
    data = fetch_data(ticker, timeframe)
    data.reset_index(inplace=True)

    # Convert Date column to string format for consistency.
    if isinstance(data['Date'].dtype, pd.DatetimeTZDtype):
        data['Date'] = data['Date'].dt.tz_localize(None)
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')

    # Step 12: Preparing the sequences and scale features for model predictions.
    features = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    sequence_length = 30
    num_sequences = len(features_scaled) - sequence_length
    sequence_list = [
        features_scaled[i:i + sequence_length]
        for i in range(num_sequences)
    ]
    sequences_np = np.array(sequence_list)
    sequences = torch.tensor(sequences_np, dtype=torch.float32)

    # Step 13: Use the LSTM model to predict the close price for each sequence.
    predictions = []
    with torch.no_grad():
        for seq in sequences:
            seq = seq.unsqueeze(0)
            pred = model(seq).item()
            predictions.append(pred)

    # Inverse transform the predictions back to the original scale.
    predicted_close = scaler.inverse_transform(
        [[0, 0, 0, pred, 0] for pred in predictions]
    )[:, 3]
    predicted_close = np.round(predicted_close, 2)

    # Step 14: Fill up the dataframe with the newly predicted values from the model.
    data['Open'] = data['Open'].round(2)
    data['Close'] = data['Close'].round(2)
    data['Predicted Close'] = [None] * sequence_length + predicted_close.tolist()
    data['Predicted Close'] = data['Predicted Close'].fillna('')

    # Ensure all datetime columns are formatted as strings.
    datetime_cols = data.select_dtypes(include=['datetimetz', 'datetime']).columns
    for col in datetime_cols:
        if isinstance(data[col].dtype, pd.DatetimeTZDtype):
            data[col] = data[col].dt.tz_localize(None)
        data[col] = data[col].dt.strftime('%Y-%m-%d')

    # Step 15: Save the dataframe in the form of an excel .xlsx file.
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    data.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.close()
    output.seek(0)

    return dcc.send_bytes(output.read(), filename=f"{ticker}_{timeframe}.xlsx")

if __name__ == '__main__':
    app.run_server(debug=False)
