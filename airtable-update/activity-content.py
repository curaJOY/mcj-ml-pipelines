import requests
from azure_openai import chat_with_openai
import config
import time

# Replace with your API key, base/table IDs, and field IDs
API_KEY = config.params['api_key']
BASE_ID = config.params['base_id']
AZURE_KEY = config.params['azure_key']
TABLE_ID_OR_NAME = config.params['activity_id']
INPUT_FIELD_ID = config.params['activity_name']
OUTPUT_FIELD_ID = config.params['activity_content']
PAGE_SIZE = 20
OFFSET = None
TEMPLATE = config.template['activity_content']
delay_between_openai_calls = 3
headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

# Fetch all records from the table
while True:
    list_records_url = f'https://api.airtable.com/v0/{BASE_ID}/{TABLE_ID_OR_NAME}?pageSize={PAGE_SIZE}'
    
    if OFFSET:
        list_records_url += f"&offset={OFFSET}"
        
    print("=== Current Offset: " + str(OFFSET) + " ===")

    response = requests.get(list_records_url, headers=headers)

    if response.status_code == 200:
        records = response.json()['records']
        updated_records = []

        # Process each record
        for record in records:
            record_id = record['id']
            input_value = record['fields'][INPUT_FIELD_ID]
            
            print("Current work: " + input_value)

            # Generate description using chat_with_openai function
            generated_description = chat_with_openai(AZURE_KEY, user_template=TEMPLATE, user_value=input_value)

            # Prepare update payload
            update_payload = {
                'id': record_id,
                'fields': {OUTPUT_FIELD_ID: generated_description}
            }
            
            #print(generated_description)

            # Add the updated record to the list
            updated_records.append(update_payload)
            
            # Introduce a delay before the next OpenAI API call
            time.sleep(delay_between_openai_calls)

        # If there are any updates to make
        if updated_records:
            # Update records in batches of 10 (Airtable limit)
            for i in range(0, len(updated_records), 10):
                batch = updated_records[i:i+10]

                # Prepare the update request body
                update_url = f'https://api.airtable.com/v0/{BASE_ID}/{TABLE_ID_OR_NAME}'
                data = {'records': batch}
                update_response = requests.patch(update_url, headers=headers, json=data)

                if update_response.status_code != 200:
                    print(f"Error updating records: {update_response.text}")
                else:
                    print(f"Successfully updated batch starting from index {i}")

        # Check for more records
        OFFSET = response.json().get('offset')
        if not OFFSET:
            break  # No more records, exit the loop

    else:
        print(f"Error fetching records: {response.text}")
        break