import os
from dotenv import load_dotenv
import sqlite3
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.llms import HuggingFaceHub
import warnings

# Suppress DeprecationWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize environment
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""

# Connect to the SQL database
def connect_to_sql_database(db_uri):
    # Load database and validate schema
    db = SQLDatabase.from_uri(db_uri)
    print("Tables in Database:", db.get_table_names())
    for table in db.get_table_names():
        print(f"Columns in '{table}':", db.get_table_info(table))
    return db

# Initialize Hugging Face LLM
llm = HuggingFaceHub(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.3, "max_new_tokens": 512}
)

# Create a query chain for SQL RAG
def create_sql_rag_chain(database):
    return create_sql_query_chain(llm, database)

# Perform RAG query on the SQL database
def rag_query_sql(question, sql_chain):
    response = sql_chain.invoke({"question": question})
    return response

# Example usage
if __name__ == "__main__":
    # Path to your SQLite database file (converted from customers-100.csv)
    db_uri = "sqlite:///customers-100.db"

    # Connect to the database and validate schema
    database = connect_to_sql_database(db_uri)

    # Create a query chain
    sql_chain = create_sql_rag_chain(database)

    # Example query tailored to the customers dataset
    question = "List all customers whose email domain is 'gmail.com'?"
    result = rag_query_sql(question, sql_chain)
    
    print("Query Result:", result)
