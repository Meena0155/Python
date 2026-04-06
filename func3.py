import pandas as pd
import json
import pyodbc
import struct
import urllib
from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from sqlalchemy import create_engine, event

# --- Configuration ---
SQL_SERVER = "srvrtestms.database.windows.net"
DATABASE = "free-sql-db-dev"
ENDPOINT = "https://testmsnew.openai.azure.com/openai/v1/"
DEPLOYMENT_NAME = "gpt-4o"
OUTPUT_TABLE = "pipeline1"

# 1. Setup Authentication
credential = DefaultAzureCredential()

# Token provider for Azure OpenAI
token_provider = get_bearer_token_provider(
    credential, 
    "https://cognitiveservices.azure.com/.default"
)

# 2. Function to get a pyodbc connection (for Reading)
def get_sql_connection():
    conn_str = (
        f"Driver={{ODBC Driver 18 for SQL Server}};"
        f"Server={SQL_SERVER};"
        f"Database={DATABASE};"
        f"Encrypt=yes;"
        f"TrustServerCertificate=no;"
        f"Connection Timeout=150;"
    )
    # Get token for Azure SQL
    token_bytes = credential.get_token("https://database.windows.net/.default").token.encode("utf-16-le")
    token_struct = struct.pack(f"<I{len(token_bytes)}s", len(token_bytes), token_bytes)
    
    # Apply token to connection
    conn = pyodbc.connect(conn_str, attrs_before={1256: token_struct})
    return conn

# 3. Load Data from Source SQL Table
print("Connecting to SQL to read source data...")
query = """
SELECT id, text, category, sub_category, home, date 
FROM [dbo].[sample_feedback] 
WHERE date >= '2025-10-01' AND date < '2025-10-08' 
AND home = 'Green Oaks'
"""

try:
    with get_sql_connection() as conn:
        df = pd.read_sql(query, conn)
    print(f"Successfully read {len(df)} records from sample_feedback.")
except Exception as e:
    print(f"Error reading from SQL: {e}")
    exit()

# Standardizing column names
df.columns = [c.lower() for c in df.columns]
records_to_process = df.to_dict(orient='records')

# 4. Load Prompt Template from Local Folder
try:
    with open("prompt.md", "r", encoding="utf-8") as f:
        prompt_markdown = f.read()
except FileNotFoundError:
    print("Error: prompt.md not found in the current directory.")
    exit()

# 5. Initialize OpenAI Client
client = OpenAI(
    base_url=ENDPOINT,
    api_key=token_provider()
)

# 6. Process in Batches and Invoke LLM
all_llm_results = []
batch_size = 3

print("Starting LLM processing...")
for i in range(0, len(records_to_process), batch_size):
    batch = records_to_process[i : i + batch_size]
    
    for record in batch:
        try:
            # Ensure the date is serializable for JSON
            if 'date' in record and hasattr(record['date'], 'isoformat'):
                record['date'] = record['date'].isoformat()

            combined_user_payload = f"{prompt_markdown}\n\n<record>\n{json.dumps(record)}\n</record>"
            
            completion = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=[{"role": "user", "content": combined_user_payload}],
                response_format={"type": "json_object"}
            )
            
            response_content = completion.choices[0].message.content
            all_llm_results.append(json.loads(response_content))
            
        except Exception as e:
            print(f"Error processing ID {record.get('id')}: {e}")

# 7. Final Result Output - Write to pipeline1 Table
if all_llm_results:
    output_df = pd.DataFrame(all_llm_results)
    
    # Prepare SQLAlchemy engine for writing
    params = urllib.parse.quote_plus(
        f"Driver={{ODBC Driver 18 for SQL Server}};"
        f"Server={SQL_SERVER};"
        f"Database={DATABASE};"
        f"Encrypt=yes;"
        f"TrustServerCertificate=no;"
    )
    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

    # Inject Managed Identity Token into SQLAlchemy connections
    @event.listens_for(engine, "do_connect")
    def provide_token(dialect, conn_rec, cargs, cparams):
        raw_token = credential.get_token("https://database.windows.net/.default").token.encode("utf-16-le")
        token_struct = struct.pack(f"<I{len(raw_token)}s", len(raw_token), raw_token)
        # Add token to connection attributes
        cparams["attrs_before"] = {1256: token_struct}

    print(f"Writing results to table {OUTPUT_TABLE}...")
    try:
        output_df.to_sql(
            name=OUTPUT_TABLE, 
            con=engine, 
            index=False, 
            if_exists='append', # Change to 'replace' if you want to overwrite the table
            schema='raw'
        )
        print(f"Success! Processed data saved to [dbo].[{OUTPUT_TABLE}].")
    except Exception as e:
        print(f"Failed to write to SQL: {e}")
else:
    print("No records were processed.")