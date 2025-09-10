import pandas as pd

def preprocess_data(data):
    """Preprocess the raw data"""
    # Remove duplicates
    data = data.drop_duplicates()
    
    # Handle missing values
    data = data.fillna(method='ffill')
    data = data.fillna(method='bfill')
    
    # Convert date column
    data['created'] = pd.to_datetime(data['created'])
    data = data.sort_values('created')
    
    # Create route column
    data['route'] = data['city_dep_name'] + ' - ' + data['city_arr_name']
    
    return data

def get_top_routes(data, n=10):
    """Get top n routes by frequency"""
    return data['route'].value_counts().nlargest(n).index.tolist()

def prepare_route_data(data, route):
    """Prepare data for a specific route"""
    df_route = data[data['route'] == route][['created', 'amount_total_sum']].copy()
    df_route.set_index('created', inplace=True)
    df_route = df_route.resample('D').min().fillna(method='ffill')
    return df_route