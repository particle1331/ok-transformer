import json
import requests


ride = {
    'PULocationID': 130,
    'DOLocationID': 205,
    'trip_distance': 3.66,
}


if __name__ == "__main__":
    
    host = 'http://0.0.0.0:9696'
    url = f'{host}/predict'
    
    response = requests.post(url, json=ride)
    result = response.json()
    
    print(result)
