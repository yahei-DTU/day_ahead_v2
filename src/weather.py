import requests

# DMI OGC Features API endpoint for meteorological observations
url = "https://dmigw.govcloud.dk/v2/metObs/collections/observation/items"
api_key = "INSERT_YOUR_API_KEY_HERE"

# Build request with proper API key (query parameter method)
params = {
    "api-key": api_key,
    # You can add filters, e.g. 'limit', 'bbox', 'datetime', 'observedProperty'
    "limit": 10
}

response = requests.get(url, params=params)
data = response.json()

print(data)  # The response is a GeoJSON FeatureCollection:
# data['features'] contains a list of observations
for feature in data['features']:
    print(feature['properties'])