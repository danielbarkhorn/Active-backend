import os
import dataset
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/restart', methods=['GET'])
def restart():
    iris = dataset.Dataset()
    iris_labeled = iris.getLabeledData()

    response = jsonify(iris_labeled)
    response.status_code = 200

    return (response)

@app.route('/label', methods=['POST'])
def label():
    iris = dataset.Dataset()
    iris_labeled = iris.getLabeledData()

    response = jsonify(iris_labeled)
    response.status_code = 200

    return (response)