import pandas as pd
import sqlite3

# Load CSV into a DataFrame
csv_file_path = r"C:\Users\Asarv\Downloads\customers-100.csv"
df = pd.read_csv(csv_file_path)

# Connect to SQLite database (creates a new database if not exists)
db_path = r"C:\Users\Asarv\Downloads\customers-100.db"
conn = sqlite3.connect(db_path)

# Save DataFrame to SQLite table
df.to_sql("customers", conn, if_exists="replace", index=False)

print("Database created successfully!")
