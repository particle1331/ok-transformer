import os

from ride_duration.predict import load_model, make_prediction
from flask import Flask, request, jsonify
from pymongo import MongoClient


EVIDENTLY_SERVICE_ADDRESS = os.getenv('EVIDENTLY_SERVICE', 'http://127.0.0.1:5000')
MONGODB_ADDRESS = os.getenv('MONGODB_ADDRESS', 'mongodb://127.0.0.1:27017')
RUN_ID = os.getenv('RUN_ID', 'f4e2242a53a3410d89c061d1958ae70a')
EXPERIMENT_ID = os.getenv('EXPERIMENT_ID', '1')

model = load_model(run_id=RUN_ID, experiment_id=EXPERIMENT_ID)
app = Flask('duration-prediction')

mongo_client = MongoClient(MONGODB_ADDRESS)
db = mongo_client.get_database("prediction-service")
collection = db.get_collection("data")


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """Predict duration of a single ride using NYCRideDurationModel."""
    
    record = request.get_json()
    pred = float(make_prediction(model, [record])[0])
    result = {
        'duration': pred,
        'model_version': RUN_ID,
    }

    save_to_db(record, pred)
    send_to_evidently_service(record, pred)

    return jsonify(result)


def save_to_db(record, prediction):
    rec = record.copy()
    rec['prediction'] = prediction
    collection.insert_one(rec)


def send_to_evidently_service(record, prediction):
    rec = record.copy()
    rec['prediction'] = prediction
    requests.post(f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/taxi", json=[rec])


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
