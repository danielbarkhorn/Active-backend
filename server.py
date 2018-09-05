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
        numLabeled = int(payload['numLabeled'])
        iris = dataset.Dataset()
        newLabels = iris.labelData(payload, numLabeled)
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

@app.route('/labelAndTest', methods=['POST'])
def labelAndTest():
    try:
        payload = request.get_json()
        numLabeled = int(payload['numLabeled'])
        iris = dataset.Dataset()
        response = iris.labelData(payload, numLabeled)
        iris.loadPayload(payload)

        svmModel = model.Model()
        svmModel.fit(iris.get_X(), iris.get_Y())

        results = svmModel.test(payload['test_X'], payload['test_Y'], target_names=iris.labels)
        response['results'] = results

        response = jsonify(response)
        response.status_code = 200

    except Exception.message:
        response = 'failure?'

    return (response)

@app.route("/")
def index():
    return "Hello World!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, threaded=True)
