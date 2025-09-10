DB_CONFIG = {
    'host': '127.0.0.1',
    'port': '5432',
    'database': 'avia',
    'user': 'postgres',
    'password': '1111'
}

def get_engine():
    from sqlalchemy import create_engine
    return create_engine(
        f'postgresql+psycopg2://{DB_CONFIG["user"]}:{DB_CONFIG["password"]}@{DB_CONFIG["host"]}:{DB_CONFIG["port"]}/{DB_CONFIG["database"]}'
    )