import os
from psycopg2.pool import ThreadedConnectionPool
from dotenv import load_dotenv
from psycopg2.pool import PoolError

# Load environment variables from a .env file (if present)
load_dotenv()

# Initialize a connection pool
connection_pool = None

def init_connection_pool():
    global connection_pool
    if connection_pool is None:
        connection_pool = ThreadedConnectionPool(
            1,  # Minimum number of connections in the pool
            10,  # Maximum number of connections in the pool
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )

def get_connection():
    global connection_pool
    if connection_pool is None:
        raise Exception("Connection pool is not initialized. Call init_connection_pool first.")
    return connection_pool.getconn()

def release_connection(connection):
    global connection_pool
    if connection_pool:
        connection_pool.putconn(connection)

def close_all_connections():
    if connection_pool:
        try:
            connection_pool.closeall()
        except PoolError as e:
            print(f"Error closing connection pool: {e}")
