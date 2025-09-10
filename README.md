# Airfare Price Prediction with LSTM

This project implements an LSTM-based time series forecasting model to predict airfare prices for the most popular flight routes. The system processes historical flight data, trains prediction models, and stores results in a PostgreSQL database.

## Features

- **Data preprocessing and cleaning** - Handling missing values and data normalization
- **Identification of top flight routes** - Automatic selection of the 10 most popular routes
- **LSTM-based time series forecasting** - Deep learning model for accurate price predictions
- **Performance metrics calculation** - RMSE, MAE, and R² evaluation metrics
- **Database integration** - PostgreSQL storage for all data and results
- **Visualization** - Graphical comparison of historical data and predictions

## Workflow

The script executes the following steps:

1. **Load and preprocess the data** - Clean and normalize historical flight data
2. **Identify top 10 flight routes** - Select routes with highest traffic
3. **Train LSTM models** - Individual model for each route
4. **Generate 5-day forecasts** - Predict prices for the next 5 days
5. **Calculate performance metrics** - Evaluate model accuracy
6. **Store results in database** - Persist predictions and metrics
7. **Display visualizations** - Generate comparison charts for each route

## Model Architecture

The LSTM model uses the following configuration:

- **Single LSTM layer** with 50 units
- **Dense output layer** with 5 units (for 5-day forecast)
- **Adam optimizer** with mean squared error (MSE) loss function
- **MinMax scaling** for data normalization (0-1 range)
- **Sequence length** of 10 days for training input
- **Batch size** of 32 for training
- **Early stopping** to prevent overfitting

## Database Schema

The project uses PostgreSQL with the following tables:

### routes
- `route_id` (PK) - Unique route identifier
- `origin` - Departure airport code
- `destination` - Arrival airport code
- `popularity_rank` - Route popularity ranking

### flights
- `flight_id` (PK) - Unique flight identifier
- `route_id` (FK) - Reference to routes table
- `airline` - Airline carrier code
- `departure_time` - Scheduled departure time

### ticket_prices
- `price_id` (PK) - Unique price record identifier
- `flight_id` (FK) - Reference to flights table
- `date` - Date of price recording
- `price` - Ticket price in local currency

### price_predictions
- `prediction_id` (PK) - Unique prediction identifier
- `route_id` (FK) - Reference to routes table
- `prediction_date` - Date when prediction was made
- `forecast_dates` - Array of dates for forecast period
- `predicted_prices` - Array of predicted prices
- `rmse` - Root Mean Square Error metric
- `mae` - Mean Absolute Error metric
- `r2_score` - R-squared coefficient of determination

## Output

The model generates:
- Forecasted prices for the next 5 days for each route
- Performance metrics (normalized RMSE, MAE, and R²)
- Visualization plots comparing historical data and predictions
- Database records with all predictions and metrics

## Dependencies

- **Python 3.7+**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning utilities and metrics
- **tensorflow** - Deep learning framework
- **matplotlib** - Data visualization
- **sqlalchemy** - Database ORM
- **psycopg2-binary** - PostgreSQL adapter
- **openpyxl** - Excel file support
