from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta, datetime


DeploymentSpec(
    name="deploy-mlflow-staging",
    flow_name='mlflow-staging',
    schedule=IntervalSchedule(interval=timedelta(minutes=1)),
    flow_location="./main.py",
    flow_storage="48da367b-1262-4d6d-ae19-38f7b706818e",
    flow_runner=SubprocessFlowRunner(),
    parameters={
        "train_data_path": './data/green_tripdata_2021-01.parquet',
        "valid_data_path": './data/green_tripdata_2021-02.parquet',
        "num_xgb_runs": 10,
        "datetime": str(datetime.now())
    },
    tags=["ml"]
)
