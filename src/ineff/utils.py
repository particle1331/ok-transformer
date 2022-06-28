import os
from ineff.config import config


def download_kaggle_dataset(dataset: str):
    """Here dataset is the portion of the URL https://www.kaggle.com/datasets/{dataset}."""
    
    user, dataset = dataset.split("/")
    data_path = config.DATASET_DIR / dataset
    if data_path.exists():
        print(f"Dataset already exists in {data_path}")
        print("Skipping download.")
    else:
        os.system(f"kaggle datasets download -d {user + '/' + dataset} -p {config.DATASET_DIR}")
        os.system(f"unzip {data_path}.zip -d {data_path} > /dev/null")
        os.system(f"rm {data_path}.zip")


def download_kaggle_competition(competition: str):
    """Here dataset is the portion of the URL https://www.kaggle.com/c/{competition}."""
    
    data_path = config.DATASET_DIR / competition
    if data_path.exists():
        print(f"Dataset already exists in {data_path}")
        print("Skipping download.")
    else:
        os.system(f"kaggle competitions download -c {competition} -p {config.DATASET_DIR}")
        os.system(f"unzip {data_path}.zip -d {data_path} > /dev/null")
        os.system(f"rm {data_path}.zip")
