import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

from config.database import get_engine
from data.data_loader import load_data, save_to_database
from utils.preprocessor import preprocess_data, get_top_routes, prepare_route_data
from models.lstm_model import create_lstm_model, create_sequences, create_train_data
from utils.metrics import calculate_metrics

# Constants
SEQ_LENGTH = 10
FORECAST_HORIZON = 5
DATA_FILE_PATH = 'E:\Семестр 8\Технологическая практика\df (пример).xlsx'

def main():
    # Initialize database connection
    engine = get_engine()
    
    # Load and preprocess data
    data = load_data(DATA_FILE_PATH)
    data = preprocess_data(data)
    
    # Get top routes
    top_routes = get_top_routes(data)
    data_top_routes = data[data['route'].isin(top_routes)]
    
    # Initialize storage for results
    metrics = {}
    predictions = []
    predictions_date = []
    
    # Process each route
    for route in top_routes:
        print(f"Processing route: {route}")
        
        # Prepare route data
        df_route = prepare_route_data(data_top_routes, route)
        
        # Scale data
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df_route)
        
        # Create sequences
        X_all = create_sequences(df_scaled, SEQ_LENGTH)
        
        # Prepare the last sequence for forecasting
        last_sequence = df_scaled[-SEQ_LENGTH:]
        input_seq = last_sequence.reshape((1, SEQ_LENGTH, 1))
        
        # Create training data
        X, y = create_train_data(df_scaled, SEQ_LENGTH, FORECAST_HORIZON)
        
        # Split data
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Reshape data for LSTM
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Create and train model
        model = create_lstm_model(SEQ_LENGTH, FORECAST_HORIZON)
        model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)
        
        # Make predictions
        y_pred = model.predict(X_test, verbose=0)
        future_pred = model.predict(input_seq, verbose=0)
        
        # Inverse transform predictions
        y_test_inv = scaler.inverse_transform(
            np.concatenate([y_test, np.zeros((y_test.shape[0], 1))], axis=1)
        )[:, :FORECAST_HORIZON]
        
        y_pred_inv = scaler.inverse_transform(
            np.concatenate([y_pred, np.zeros((y_pred.shape[0], 1))], axis=1)
        )[:, :FORECAST_HORIZON]
        
        future_pred_inv = scaler.inverse_transform(
            np.concatenate([future_pred, np.zeros((future_pred.shape[0], 1))], axis=1)
        )[:, :FORECAST_HORIZON]
        
        predictions.append(future_pred_inv)
        
        # Calculate metrics
        data_range = df_route.values.max() - df_route.values.min()
        route_metrics = calculate_metrics(y_test_inv, y_pred_inv, data_range)
        metrics[route] = route_metrics
        
        # Prepare for visualization
        future_dates = pd.date_range(
            start=df_route.index[-1] + pd.Timedelta(days=1), 
            periods=FORECAST_HORIZON
        )
        predicted_series = pd.Series(future_pred_inv.flatten(), index=future_dates)
        predictions_date.append(future_dates)
        
        # Print metrics
        print(f'RMSE: {route_metrics["RMSE"]:.3f}')
        print(f'MAE: {route_metrics["MAE"]:.3f}')
        
        # Create visualization
        plt.figure(figsize=(12, 5))
        plt.plot(df_route.index[-50:], df_route.values[-50:], label='Historical Data')
        plt.plot(
            pd.DatetimeIndex([df_route.index[-1].strftime("%Y-%m-%d %H:%M:%S")]).append(
                predicted_series.index
            ), 
            np.concatenate([df_route.values[-1], predicted_series.values]), 
            label='Forecast'
        )
        plt.title(f'{route}')
        plt.xlabel('Purchase Date')
        plt.ylabel('Ticket Price')
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=-90)
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # Save results to database
    save_results_to_db(data, top_routes, predictions, predictions_date, engine)
    
def save_results_to_db(data, top_routes, predictions, predictions_date, engine):
    """Save all results to database"""
    # Save routes
    routes = pd.DataFrame({
        'route': data['route'].value_counts().index.tolist()
    })
    save_to_database(routes, 'routes', engine)
    
    # Save flights
    route_mapping = {route: idx + 1 for idx, route in enumerate(routes['route'])}
    flights = pd.DataFrame({
        'route_id': data['route'].map(route_mapping),
        'departure_city': data['city_dep_name'],
        'arrival_city': data['city_arr_name'],
        'airline': data['carrier_code'],
        'aircraft': data['aircraft'],
        'flight_duration': data['flight_duration'],
        'status': data['status_id']
    })
    save_to_database(flights, 'flights', engine)
    
    # Save ticket prices
    ticket_prices = pd.DataFrame({
        'flight_id': range(1, len(flights) + 1),
        'fare_sum': data['fare_sum'],
        'service_fee_sum': data['service_fee_sum'],
        'tax_sum': data['tax_sum'],
        'total_sum': data['amount_total_sum'],
        'baggage': data['baggage'],
        'created_at': data['created']
    })
    save_to_database(ticket_prices, 'ticket_prices', engine)
    
    # Save predictions
    preds = pd.DataFrame({
        'route_id': np.array([[i] * 5 for i in range(1, len(top_routes) + 1)]).flatten().tolist(),
        'predicted_price': np.array(predictions).flatten().tolist(),
        'prediction_date': pd.to_datetime(np.array(predictions_date).flatten().tolist()).strftime('%Y-%m-%d %H:%M:%S'),
        'created_at': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] * FORECAST_HORIZON * len(top_routes)
    })
    save_to_database(preds, 'price_predictions', engine)

if __name__ == "__main__":
    main()