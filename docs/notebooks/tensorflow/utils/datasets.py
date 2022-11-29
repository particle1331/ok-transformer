import sys
from typing import Dict, Optional, Callable
import urllib.request

import numpy as np
import tensorflow as tf
from pathlib import Path

DATASET_DIR = Path("../input/").absolute()


class MNIST:
    H: int = 28
    W: int = 28
    C: int = 1
    LABELS: int = 10

    def __init__(self, 
            batch_size: int,
            split_sizes: Dict[str, int] = {},
            seed: int = 42,
            train_transforms: Optional[Callable] = None,
            infer_transforms: Optional[Callable] = None
        ) -> None:

        identity = lambda image, label: (image, label)
        train_transforms = identity if train_transforms is None else train_transforms
        infer_transforms = identity if infer_transforms is None else infer_transforms

        mnist_npz = self.load_data()
        for split in ["train", "dev", "test"]:
            data = {}
            data[f"images"] = mnist_npz[f"{split}_images"][:split_sizes.get(split, None)]
            data[f"labels"] = mnist_npz[f"{split}_labels"][:split_sizes.get(split, None)]
            shuffle = (split == "train")
            transforms = train_transforms if (split == "train") else infer_transforms
            dataloader = self.dataloader(
                data=data,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed,
                transforms=transforms
            )
            setattr(self, split, dataloader)

    @staticmethod
    def load_data():
        url  = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/2122/datasets/"
        path = "mnist.npz"
        local_path = DATASET_DIR / path

        if not local_path.exists():
            print("Downloading dataset mnist...", file=sys.stderr)
            urllib.request.urlretrieve(f"{url}/{path}", filename=local_path)

        mnist_npz = np.load(local_path)
        return mnist_npz

    @staticmethod
    def dataloader(data, batch_size, shuffle, seed, transforms):
        ds = tf.data.Dataset.from_tensor_slices((data["images"], data["labels"]))
        ds = ds.shuffle(int(0.10 * len(ds)), seed=seed) if shuffle else ds
        ds = ds.batch(batch_size, drop_remainder=True).map(transforms)
        ds = ds.prefetch(tf.data.AUTOTUNE)  # dynamically adjust no. of threads
        return ds
