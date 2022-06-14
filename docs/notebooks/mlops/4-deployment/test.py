import joblib
from train import AddPickupDropoffPair, SelectFeatures, ConvertToString, ConvertToDict
from predict import predict


ride = [{
    'VendorID': 2,
    'store_and_fwd_flag': 'N',
    'RatecodeID': 1.0,
    'PULocationID': 130,
    'DOLocationID': 205,
    'passenger_count': 5.0,
    'trip_distance': 3.66,
    'fare_amount': 14.0,
    'extra': 0.5,
    'mta_tax': 0.5,
    'tip_amount': 10.0,
    'tolls_amount': 0.0,
    'ehail_fee': None,
    'improvement_surcharge': 0.3,
    'total_amount': 25.3,
    'payment_type': 1.0,
    'trip_type': 1.0,
    'congestion_surcharge': 0.0
}]


if __name__ == "__main__":
    print(predict(ride))
