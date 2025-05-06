import pandas as pd
import numpy as np
import os
from ..base import EmbeddingStorageStrategy

class ParquetStorageStrategy(EmbeddingStorageStrategy):
    def __init__(self, path):
        self.path = path
    def save(self, embeddings):
        if isinstance(embeddings, pd.DataFrame):
            embeddings.to_parquet(self.path)
        