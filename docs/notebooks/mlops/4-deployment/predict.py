import joblib
import pandas as pd
from train import AddPickupDropoffPair, SelectFeatures, ConvertToString, ConvertToDict
from flask import Flask, request, jsonify

model = joblib.load('models/lin_reg.bin')


def predict(features: list[dict]):

    X = pd.DataFrame(features)
    preds = model.predict(X)
    return preds[0]
