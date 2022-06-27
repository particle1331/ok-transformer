from datetime import datetime
from dateutil.relativedelta import relativedelta

from prefect import flow

import score


@flow
def ride_duration_prediction_backfill(
    run_id: str, 
    experiment_id: str,
    taxi_type: str, 
    start_date: datetime, 
    end_date: datetime
):
    """Run batch scoring flows for run dates 
    between start_date and end_date (inclusive)."""

    run_date = start_date

    while run_date <= end_date:

        score.ride_duration_prediction(
            taxi_type=taxi_type,
            experiment_id=experiment_id,
            run_id=run_id,
            run_date=run_date
        )
        
        run_date = run_date + relativedelta(months=1)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxi-type", type=str)
    parser.add_argument("--experiment-id", type=str)
    parser.add_argument("--run-id", type=str)
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--start-month", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--end-month", type=int)
    args = parser.parse_args()
    
    start_date = datetime(year=args.start_year, month=args.start_month, day=1)
    end_date   = datetime(year=args.end_year,   month=args.end_month,   day=1)

    ride_duration_prediction_backfill(
        experiment_id=args.experiment_id,
        run_id=args.run_id,
        taxi_type=args.taxi_type,
        start_date=start_date, 
        end_date=end_date
    )
