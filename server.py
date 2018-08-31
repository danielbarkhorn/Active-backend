import os
import dataset
from flask import Flask, jsonify, request
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

@app.route('/restart', methods=['GET'])
def restart():
    iris = dataset.Dataset()
    iris_json = iris.createRandomSampling()

    response = jsonify(iris_json)
    response.status_code = 200

    return (response)

@app.route('/label', methods=['POST'])
def label():
    try:
        payload = request.get_json()
        iris = dataset.Dataset()
        newLabels = iris.labelData(payload)
        response = jsonify(newLabels)
        response.status_code = 200

    except Exception.message:
        response = 'failure?'

    return (response)
