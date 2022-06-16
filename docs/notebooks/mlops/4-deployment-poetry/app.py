import joblib
import pandas as pd

from ride_duration.predict import load_model, make_prediction, prepare_features
from flask import Flask, request, jsonify


model = load_model()
app = Flask('duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()
    preds = make_prediction(model, ride)

    result = {
        'duration': float(preds[0])
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
