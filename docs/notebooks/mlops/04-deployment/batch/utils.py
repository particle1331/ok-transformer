import uuid

from dateutil.relativedelta import relativedelta


def save_results(df, preds, run_id, output_file):
    """Save output dataframe containing model predictions."""

    results_cols = [
        'lpep_pickup_datetime', 
        'PULocationID', 
        'DOLocationID', 
        'trip_distance', 
        'duration'
    ]
    df_results = df[results_cols].copy()
    
    df_results['model_version']= run_id
    df_results['actual_duration'] = df['duration']
    df_results['predicted_duration'] = preds
    df_results['diff'] = df_results['actual_duration'] - df_results['predicted_duration']
    df_results['ride_id'] = [str(uuid.uuid4()) for _ in range(len(df))]

    # Saving results
    df_results.to_parquet(output_file, index=False)


def get_paths(run_date, taxi_type, run_id):
    """Get input and output file paths from scheduled date."""

    # Get previous month and year from run date
    # e.g. run date=2021/06/02 -> month=5, year=2021 (input).
    prev_month = run_date - relativedelta(months=1)
    year = prev_month.year
    month = prev_month.month 

    input_file = (
        f's3://nyc-tlc/trip data/'
        f'{taxi_type}_tripdata_'
        f'{year:04d}-{month:02d}.parquet'
    )
    
    output_file = (
        f's3://nyc-duration-prediction-ron/' 
        f'taxi_type={taxi_type}/'
        f'year={year:04d}/'
        f'month={month:02d}/'
        f'{run_id}.parquet'
    )

    return input_file, output_file