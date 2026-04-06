#read data from blob and write into blob

import pandas as pd
import json
import io
from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.storage.blob import BlobServiceClient

# --- Configuration ---
ACCOUNT_URL = "https://testmssrc.blob.core.windows.net"
CONTAINER_NAME = "input"
ENDPOINT = "https://testmsfoundrynew.openai.azure.com/openai/v1/"
DEPLOYMENT_NAME = "gpt-4o"
OUTPUT_BLOB_NAME = "output.csv"

# 1. Setup Authentication & Clients
credential = DefaultAzureCredential()
# Bearer token provider for Azure OpenAI / Foundry
token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")

blob_service_client = BlobServiceClient(ACCOUNT_URL, credential=credential)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)

# OpenAI Client initialized with Entra ID token
client = OpenAI(
    base_url=ENDPOINT,
    api_key=token_provider() # Get initial token
)

def get_blob_content(blob_name):
    return container_client.get_blob_client(blob_name).download_blob().readall()

# 2. Load and Filter Data
csv_data = get_blob_content("sample_feedback.csv")
df = pd.read_csv(io.BytesIO(csv_data))

# Standardizing column names to lowercase for consistency
df.columns = [c.lower() for c in df.columns]

# Filter: Date between 2025-10-01 and 2025-10-08 for Green Oaks
df['date'] = pd.to_datetime(df['date'])
mask = (df['date'] >= '2025-10-01') & (df['date'] < '2025-10-08') & (df['home'] == 'Green Oaks')
filtered_df = df.loc[mask, ['id', 'text', 'category', 'sub_category', 'home']]

records_to_process = filtered_df.to_dict(orient='records')

# 3. Load Prompt Template
prompt_markdown = get_blob_content("prompt.md").decode('utf-8')

# 4. Process in Batches and Invoke LLM
all_llm_results = []
batch_size = 3

for i in range(0, len(records_to_process), batch_size):
    batch = records_to_process[i : i + batch_size]
    
    for record in batch:
        try:
            # Refresh token for every call or batch if necessary 
            # (token_provider is called inside the client for most modern implementations, 
            # but we can pass the fresh key here if needed)
            
            # Combine Prompt and Data into a single User message with delimiters
            combined_user_payload = f"{prompt_markdown}\n\n<record>\n{json.dumps(record)}\n</record>"
            
            completion = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=[
                    {
                        "role": "user", 
                        "content": combined_user_payload
                    }
                ],
                response_format={"type": "json_object"}
            )
            
            # Parse the JSON response
            response_content = completion.choices[0].message.content
            all_llm_results.append(json.loads(response_content))
            
        except Exception as e:
            # Note: used lowercase 'id' to match the standardized columns
            print(f"Error processing ID {record.get('id')}: {e}")

# 5. Convert Results to CSV and Upload
if all_llm_results:
    output_df = pd.DataFrame(all_llm_results)
    
    output_buffer = io.StringIO()
    output_df.to_csv(output_buffer, index=False)
    
    output_blob_client = container_client.get_blob_client(OUTPUT_BLOB_NAME)
    output_blob_client.upload_blob(output_buffer.getvalue(), overwrite=True)
    
    print(f"Successfully processed {len(all_llm_results)} records to {OUTPUT_BLOB_NAME}")
else:
    print("No records matched the criteria or were processed.")