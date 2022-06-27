import os
import sys

import pandas as pd
import mlflow

from prefect import task, flow, get_run_logger
from prefect.context import get_run_context

from ride_duration.utils import load_training_dataframe
from ride_duration.predict import load_model, make_prediction
from utils import save_results, get_paths

from datetime import datetime


@task
def apply_model(input_file, output_file, experiment_id, run_id):
    """Load input and model. Make predictions on the input file."""
    
    # Get prefect logger
    logger = get_run_logger()

    logger.info(f'Reading the data from {input_file}')
    df = load_training_dataframe(input_file)

    logger.info(f'Loading model {experiment_id}/{run_id}')
    model = load_model(experiment_id, run_id)

    logger.info(f'Applying the model')
    y_pred = make_prediction(model, df)

    logger.info(f'Saving the result to {output_file}')
    save_results(df, y_pred, run_id, output_file)


@flow
def ride_duration_prediction(
        taxi_type: str,
        run_id: str,
        experiment_id: str,
        run_date: datetime = None
    ) -> None:

    # Get scheduled data if no run_date
    if run_date is None:
        ctx = get_run_context()
        run_date = ctx.flow_run.expected_start_time
    
    # Get input path and output path in S3
    input_file, output_file = get_paths(run_date, taxi_type, run_id)

    # Execute make predictions on input task
    apply_model(
        input_file=input_file,
        output_file=output_file,
        run_id=run_id,
        experiment_id=experiment_id
    )


if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxi-type", type=str)
    parser.add_argument("--experiment-id", type=str)
    parser.add_argument("--run-id", type=str)
    parser.add_argument("--year", type=int)
    parser.add_argument("--month", type=int)
    args = parser.parse_args()

    # Run flow
    ride_duration_prediction(
        taxi_type=args.taxi_type,
        run_id=args.run_id,
        experiment_id=args.experiment_id,
        run_date=datetime(year=args.year, month=args.month, day=2)
    )
