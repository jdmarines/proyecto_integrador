import pandas as pd
from .base import LoadDataset

class LoadWithPandas(LoadDataset):
    def __init__(self, data_path, sep):
        self.data_path = data_path
        self.sep = sep

    def load_data(self):       
        return pd.read_csv(self.data_path, sep=self.sep )
