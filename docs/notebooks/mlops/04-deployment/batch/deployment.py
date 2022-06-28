from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner


DeploymentSpec(
    name="ride-duration-prediction-monthly",
    flow_location="./score.py",
    flow_name="ride-duration-prediction",
    parameters={
        "taxi_type": "green",
        "run_id": "f4e2242a53a3410d89c061d1958ae70a",
        "experiment_id": "1",
    },
    flow_storage="68bb8db9-ae1c-4ee7-a766-a1eb3b9a8eed",
    schedule=CronSchedule(cron="0 3 2 * *"), # https://crontab.guru/#0_3_2_*_*
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"]
)
