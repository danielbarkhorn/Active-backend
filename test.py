import pandas as pd
import requests

header = {'Content-Type': 'application/json', 'Accept': 'application/json'}
jsonData = pd.read_csv('./data/iris.csv', header=None).to_json()
print(jsonData)

resp = requests.post("http://0.0.0.0:8000/predict", data = jsonData, headers=header)
print(resp.status_code)
