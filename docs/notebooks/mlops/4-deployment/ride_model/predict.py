import joblib
from ride_model.utils import package_dir


def load_model():
    return joblib.load(package_dir / 'pipeline.pkl')
