import os
import model

RUN_ID = os.getenv('RUN_ID')

model_service = model.init(run_id=RUN_ID)


def lambda_handler(event, context):
    # pylint: disable=unused-argument
    return model_service.lambda_handler(event)
