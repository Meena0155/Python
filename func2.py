import pandas as pd
import json
#import io
import pyodbc
import struct
from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# --- Configuration ---
# Update these with your specific server and database names
SQL_SERVER = "srvrtestms.database.windows.net"
DATABASE = "free-sql-db-dev"
ENDPOINT = "https://testmsnew.openai.azure.com/openai/v1/"
DEPLOYMENT_NAME = "gpt-4o"

# 1. Setup Authentication
credential = DefaultAzureCredential()

# Token provider for Azure OpenAI
token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")

# 2. Azure SQL Connection with Managed Identity (Entra ID)
def get_sql_connection():
    # Connection string for Azure SQL
    conn_str = f"Driver={{ODBC Driver 18 for SQL Server}};Server={SQL_SERVER};Database={DATABASE};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
    
    # Get token for Azure SQL
    token_bytes = credential.get_token("https://database.windows.net/.default").token.encode("utf-16-le")
    token_struct = struct.pack(f"<I{len(token_bytes)}s", len(token_bytes), token_bytes)
    
    # Apply token to connection
    conn = pyodbc.connect(conn_str, attrs_before={1256: token_struct})
    return conn

# 3. Load Data from SQL
query = """
SELECT id, text, category, sub_category, home, date 
FROM [dbo].[sample_feedback] 
WHERE date >= '2025-10-01' AND date < '2025-10-08' 
AND home = 'Green Oaks'
"""

with get_sql_connection() as conn:
    df = pd.read_sql(query, conn)

# Standardizing column names
df.columns = [c.lower() for c in df.columns]
records_to_process = df.to_dict(orient='records')

# 4. Load Prompt Template from Local Folder
with open("prompt.md", "r", encoding="utf-8") as f:
    prompt_markdown = f.read()

# 5. Initialize OpenAI Client
client = OpenAI(
    base_url=ENDPOINT,
    api_key=token_provider()
)

# 6. Process in Batches and Invoke LLM
all_llm_results = []
batch_size = 3

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

# 7. Final Result Output (Saving locally as CSV)
if all_llm_results:
    output_df = pd.DataFrame(all_llm_results)
    output_df.to_csv("processed_results.csv", index=False)
    print(f"Successfully processed {len(all_llm_results)} records to processed_results.csv")
else:
    print("No records matched the criteria.")