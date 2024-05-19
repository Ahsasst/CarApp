from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('car_price_evaluate.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return "Hello World"


@app.route('/predict', methods=['POST'])
def predict():
    categoryId = request.form.get('categoryId')
    markName = request.form.get('markName')
    modelName = request.form.get('modelName')
    year = request.form.get('year')
    raceInt = request.form.get('raceInt')
    fuelId = request.form.get('fuelId')
    gearBoxId = request.form.get('gearBoxId')
    gearBoxId = request.form.get('gearBoxId')
    custom = request.form.get('custom')
    damage = request.form.get('damage')

    input_query = np.array([[categoryId, markName, modelName, year, raceInt, fuelId, gearBoxId, custom, damage]])

    result = model.predict(input_query)[0]
    return jsonify({'evaluation': str(result)})


if __name__ == '__main__':
    app.run(debug=True)
