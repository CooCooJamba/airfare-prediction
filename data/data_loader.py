import pandas as pd
from sqlalchemy import create_engine

def load_data(file_path):
    """Load data from Excel file"""
    return pd.read_excel(file_path)

def save_to_database(df, table_name, engine, if_exists='append'):
    """Save DataFrame to database"""
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)