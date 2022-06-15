import os
import sys
import pandas as pd
import mlflow

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

from ride_duration.utils import load_training_dataframe
from ride_duration.predict import load_model, make_prediction

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def apply_model(
    input_file: str, 
    run_id: str, 
    output_file: str
) -> None:
    
    print(f'Reading the data from {input_file}...')
    df = load_training_dataframe(input_file)

    print(f'Loading the model with RUN_ID={run_id}...')
    model = load_model()

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
    df_result.to_parquet(output_file, index=False)


def run(taxi_type: str, year: int, month: int, run_id: str) -> None:

    source_url = 'https://s3.amazonaws.com/nyc-tlc/trip+data'
    input_file = f'{source_url}/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'

    apply_model(
        input_file=input_file,
        run_id=run_id,
        output_file=output_file
    )


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxi_type", default='green', type=str)
    parser.add_argument("--year", default=2021, type=int)
    parser.add_argument("--month", default=1, type=int)
    parser.add_argument("--run_id", default='e1efc53e9bd149078b0c12aeaa6365df', type=str)
    args = parser.parse_args()
    
    run(
        taxi_type=args.taxi_type,
        year=args.year,
        month=args.month,
        run_id=args.run_id
    )
