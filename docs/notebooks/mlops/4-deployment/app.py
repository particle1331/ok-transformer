
import joblib
import pandas as pd

from ride_model.preprocessing import prepare_features
from ride_model.predict import load_model
from flask import Flask, request, jsonify


model = load_model()
app = Flask('duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()
    ride = prepare_features([ride])
    preds = float(model.predict(ride)[0])

    result = {
        'duration': preds
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
