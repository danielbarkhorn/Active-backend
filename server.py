import os
import dataset
from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import model

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

@app.route('/activeSelect', methods=['POST'])
def activeSelect():
    try:
        payload = request.get_json()
        iris = dataset.Dataset()
        features = iris.features
        iris.loadPayload(payload)

        numChosen = int(payload['numChosen'])

        svmModel = model.Model()
        svmModel.fit(iris.get_X(), iris.get_Y())
        active = svmModel.activeChoice(numChosen, iris.getUnlabeled())

        chosen = {'labeled': iris.labeled, 'unlabeled': iris.unlabeledDict, 'selected': {features[i] : [a[i] for a in active] for i in range(len(features[:-1]))}}

        response = jsonify(chosen)

    except Exception.message:
        response = 'failure'

    return (response)
