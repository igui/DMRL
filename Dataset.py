from pathlib import Path
from typing import Tuple
import scipy.sparse as sp
import numpy as np
import pandas as pd

class Dataset(object):
    def __init__(self, path: Path):
        self.num_items = len(pd.read_csv(path / 'items.txt'))
        self.num_users = len(pd.read_csv(path / 'users.txt'))
        shape = (self.num_users, self.num_items)

        train_file = path / "train.txt"
        self.trainMatrix = self.load_interaction_matrix(train_file, shape)
        self.train_num = len(pd.read_csv(train_file))
        self.testRatings = self.load_interaction_matrix(path / "validation.txt", shape)

        self.imagefeatures = np.load(path / 'embed_image.npy').astype(np.float32)
        self.textualfeatures = np.load(path / 'embed_text.npy').astype(np.float32)

    def load_interaction_matrix(self, file: str, shape: Tuple[int, int]) -> sp.dok_matrix:
        df = pd.read_csv(file)
        return sp.csr_matrix(
            (np.ones_like(df['user']), (df['user'], df['item'])),
            shape=shape,
            dtype=np.float32
        )
