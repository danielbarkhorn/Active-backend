import os
import pandas as pd
from sklearn.externals import joblib
import dataset
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/restart', methods=['GET'])

def apicall():
    data = pd.read_csv('./data/iris.csv', header=None)

    responses = jsonify(predictions=data.to_json(orient="records"))
    responses.status_code = 200
    responses.header('Access-Control-Allow-Origin')

    return (responses)
