import dataset
from flask import jsonify

iris = dataset.Dataset()

print(jsonify(iris.getLabeledData()))
