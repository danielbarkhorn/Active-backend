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
    iris_labeled = iris.getLabeledData()
    iris_unlabeled = iris.getUnlabeledData()

    iris_json = {'labeled': iris_labeled, 'unlabeled': iris_unlabeled}

    response = jsonify(iris_json)
    response.status_code = 200

    return (response)

<<<<<<< Updated upstream
@app.route('/label', methods=['POST'])
def label():
    iris = dataset.Dataset()
    iris_labeled = iris.getLabeledData()

    response = jsonify(iris_labeled)
    response.status_code = 200

    return (response)
=======
# @app.route('/label', methods=['POST'])
# def label():
#     try:
#         unlabeled
>>>>>>> Stashed changes
