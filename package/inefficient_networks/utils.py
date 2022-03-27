import os
from inefficient_networks.config import config


def download_kaggle_dataset(dataset: str):
    """Here dataset is the portion of the URL https://www.kaggle.com/datasets/{dataset}."""
    os.system(f"kaggle datasets download -d {dataset} -p {config.DATASET_DIR}")
    dataset = dataset.split("/")[1]
    os.system(f"unzip {config.DATASET_DIR / dataset}.zip -d {config.DATASET_DIR / dataset}")
    os.system(f"rm {config.DATASET_DIR / dataset}.zip")


def download_kaggle_competition(competition: str):
    """Here dataset is the portion of the URL https://www.kaggle.com/c/{competition}."""
    os.system(f"kaggle competitions download -c {competition} -p {config.DATASET_DIR}")
    os.system(f"unzip {config.DATASET_DIR / competition}.zip -d {config.DATASET_DIR / competition}")
    os.system(f"rm {config.DATASET_DIR / competition}.zip")
