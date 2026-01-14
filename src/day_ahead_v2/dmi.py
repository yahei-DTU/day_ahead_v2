import sys
import requests

# Your DMI API key
api_key = "4333c624-f42e-413c-8a76-f37c4a458e34"
# Params for collections - only api key needed
collections_params = {
    "api-key": api_key,
}
# Params for items - include datetime and limit filters
items_params = {
    "api-key": api_key,
    "limit": 2000,
    # "datetime": "2025-11-15T00:00:00Z/2025-11-15T23:59:59Z"
}

# Base URL for the forecast data API
base_url = "https://dmigw.govcloud.dk/v2/metObs"

# Step 1: Access the collections endpoint to find available collections
collections_url = f"{base_url}/collections"
collections_response = requests.get(collections_url,
                                    params=collections_params,
                                    timeout=10
                                    )
collections_data = collections_response.json()

print("Available collections:")
for collection in collections_data.get('collections', []):
    print(f"- ID: {collection['id']}, Title: {collection.get('title', '')}")

# Step 2: Choose a collection ID to query forecast items
if collections_data.get('collections'):
    for collection in collections_data['collections']:
        collection_id = collection['id']

        # Step 3: Fetch forecast items for the collection
        items_url = f"{base_url}/collections/{collection_id}/items"
        items_response = requests.get(items_url, params=items_params, timeout=10)
        items_data = items_response.json()

        print(f"\nForecast items from collection '{collection_id}':")
    for feature in items_data.get('features', []):
        # Each feature's properties contain the forecast details
        properties = feature.get('properties', {})
        print(properties)

else:
    print("No collections found in API response.")

print(type(items_data.get('features', [])))
print(items_data.get('features', [])[0])
print(len(items_data.get('features', [])))
