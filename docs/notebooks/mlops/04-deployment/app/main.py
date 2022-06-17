import joblib
import pandas as pd
import os

from ride_duration.predict import load_model, make_prediction
from flask import Flask, request, jsonify


model, run_id = load_model()
app = Flask('duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """Predict ride duration using NYCRideDurationModel."""
    
    ride = request.get_json()
    preds = make_prediction(model, ride)

    return jsonify({
        'duration': float(preds[0]),
        'model_version': run_id,
    })


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
