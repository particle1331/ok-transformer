from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta, datetime


DeploymentSpec(
    name="deploy-mlflow-staging",
    flow_name='mlflow-staging',
    schedule=IntervalSchedule(interval=timedelta(minutes=1)),
    flow_location="./main.py",
    flow_storage="33133d27-a83b-468c-bb72-a9f19bd0d157",
    flow_runner=SubprocessFlowRunner(),
    parameters={
        "train_data_path": './data/green_tripdata_2021-01.parquet',
        "valid_data_path": './data/green_tripdata_2021-02.parquet',
        "num_xgb_runs": 10,
    },
    tags=["ml"]
)
