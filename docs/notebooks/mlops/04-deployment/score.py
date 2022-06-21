import os
import sys
import pandas as pd
import mlflow
import uuid

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

from ride_duration.utils import load_training_dataframe
from ride_duration.predict import load_model, make_prediction


def apply_model(
    input_file: str, 
    output_file: str, 
    run_id: str, 
    experiment_id: str
):
    
    print(f'Reading the data from {input_file}...')
    df = load_training_dataframe(input_file)
    df['ride_id'] = [str(uuid.uuid4()) for i in range(len(df))]  

    # Force download model from S3 
    model, _ = load_model(
        tracking_server_host=None, 
        run_id=run_id, 
        experiment_id=experiment_id
    )

    print(f'Applying the model...')
    preds = make_prediction(model, df)

    print(f'Saving the result to {output_file}...')
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['lpep_pickup_datetime'] = df['lpep_pickup_datetime']
    df_result['PULocationID'] = df['PULocationID']
    df_result['DOLocationID'] = df['DOLocationID']
    df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = preds
    df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']
    df_result['model_version'] = run_id
    
    return df_result


def run(
    taxi_type: str, 
    year: int, 
    month: int, 
    run_id: str, 
    experiment_id: int
):
    """Apply model on dataset parameterized for given taxi type, year, and month."""
    
    input_file = f'https://s3.amazonaws.com/nyc-tlc/trip+data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'

    result = apply_model(
        input_file=input_file,
        output_file=output_file,
        run_id=run_id,
        experiment_id=experiment_id,
    )

    result.to_parquet(output_file, index=False)


if __name__ == '__main__':
    import argparse
    import ssl
    from pathlib import Path
    from dotenv import load_dotenv

    # For some reason SSL: CERTIFICATE_VERIFY_FAILED with Pipenv
    ssl._create_default_https_context = ssl._create_unverified_context 
    
    # Loading environmental variables from .env
    load_dotenv()

    # Load command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxi_type", default='green', type=str)
    parser.add_argument("--year", default=2021, type=int)
    parser.add_argument("--month", default=1, type=int)
    parser.add_argument("--run_id", default=os.getenv("MODEL_RUN_ID"), type=str)
    parser.add_argument("--experiment_id", default=os.getenv("EXPERIMENT_ID"), type=int)    
    args = parser.parse_args()
    
    # Run batch scoring
    run(
        taxi_type=args.taxi_type,
        year=args.year,
        month=args.month,
        run_id=args.run_id,
        experiment_id=args.experiment_id
    )
