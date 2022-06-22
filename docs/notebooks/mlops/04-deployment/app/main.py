import os

from ride_duration.predict import load_model, make_prediction
from flask import Flask, request, jsonify


# Load model with run ID and experiment ID defined in the env.
RUN_ID = os.getenv("MODEL_RUN_ID")
EXPERIMENT_ID = os.getenv("EXPERIMENT_ID")
model = load_model(run_id=RUN_ID, experiment_id=EXPERIMENT_ID)

app = Flask('duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """Predict duration of a single ride using NYCRideDurationModel."""
    
    ride = request.get_json()
    preds = make_prediction(model, [ride])

    return jsonify({
        'duration': float(preds[0]),
        'model_version': RUN_ID,
    })


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
