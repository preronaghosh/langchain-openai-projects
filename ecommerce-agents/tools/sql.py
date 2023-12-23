import sqlite3
from pydantic.v1 import BaseModel
from typing import List
from langchain.tools import Tool

conn = sqlite3.connect("db.sqlite")

# Function to return the names of tables inside the database only
# for static data used as a SystemMessage inside a prompt
def list_tables():
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    rows = c.fetchall() # returns a list of tuples like [('users',), ('addresses',), ...] of all the tables
    # Format the list into a string for gpt prompt
    return "\n".join(row[0] for row in rows if row[0] is not None)


class RunQueryArgsSchema(BaseModel):
    query:str

def run_sqlite_query(query):
    c = conn.cursor()
    try:
        c.execute(query)
        return c.fetchall()
    # In case the query looks for a column that does not exist
    except sqlite3.OperationalError as err:
        return f"The following error occurred: {str(err)}"
    

run_query_tool = Tool.from_function(
    name="run_sqlite_query",
    description="Run a sqlite query.",
    func=run_sqlite_query,
    args_schema=RunQueryArgsSchema
)

class DescribeTablesArgsSchema(BaseModel):
    table_names: List[str]

# Method to take in a list of table names and formats it as a string of table names for GPT
# Example: "'users', 'addresses', 'products'" 
def describe_tables(table_names):
    c = conn.cursor()
    tables = ', '.join("'" + table + "'" for table in table_names)
    rows = c.execute(f"SELECT sql FROM sqlite_master WHERE type='table' and name IN ({tables});")
    return "\n".join(row[0] for row in rows if row[0] is not None)

describe_tables_tool = Tool.from_function(
    name="describe_tables",
    description="Given a list of table names, returns the schema of those tables.",
    func=describe_tables,
    args_schema=DescribeTablesArgsSchema
)