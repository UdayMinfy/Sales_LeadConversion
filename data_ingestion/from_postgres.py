
# Go one directory up and point to 'data/raw'
#save_path = os.path.abspath(os.path.join(os.getcwd(), "data", "raw", "LeadScoring.csv"))
import psycopg2
import pandas as pd

# Connect to your local PostgreSQL (host.docker.internal points to host from container)
conn = psycopg2.connect(
    dbname="mydb5",
    user="postgres",
    password="newpassword",  # Replace with your actual password
    host="host.docker.internal",
    port="5432"
)
#query = """
#    SELECT table_schema, table_name 
#    FROM information_schema.tables 
#    WHERE table_type='BASE TABLE' AND table_schema NOT IN ('pg_catalog', 'information_schema');
#"""
#tables = pd.read_sql(query, conn)
#print("Available tables:")
#print(tables)    
df = pd.read_sql('SELECT * FROM "LeadScoring"', conn)
df.to_csv("/opt/airflow/data/raw/LeadScoring.csv", index=False)
print("✅ Data exported to CSV successfully.")