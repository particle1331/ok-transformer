import os
from inefficient_networks.config import config


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


def plot_model_history(history, ax, metric='accuracy', label='', **kwargs):
    """Plotting result of Keras model training.
    Example:
    >>> fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    >>> plot_model_history(history=hist, ax=ax)
    """

    train_label = f'Train ({label})' if len(label) > 0 else 'Train'
    valid_label = f'Valid ({label})' if len(label) > 0 else 'Valid'
    ax[0].plot(history.history['loss'], label=train_label, color="C0", **kwargs)
    ax[0].plot(history.history['val_loss'], label=valid_label, color="C1", **kwargs)
    ax[0].set_ylabel('loss')
    ax[0].legend()

    ax[1].plot(history.history[metric], label=train_label, color="C0", **kwargs)
    ax[1].plot(history.history[f'val_{metric}'], label=valid_label, color="C1", **kwargs)
    ax[1].set_ylabel(metric)
    ax[1].legend()
