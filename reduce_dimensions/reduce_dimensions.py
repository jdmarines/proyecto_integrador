import umap
import numpy as np
import tensorflow as tf
import pandas as pd


class UmapReduction:
    def __init__(self, column, column_index, dataset, n_neighbors = 15, n_components = 2, metric = 'cosine'):
       self.dataset = dataset
       self.column = column
       self.column_index = column_index
       self.n_neighbors = n_neighbors
       self.n_components = n_components
       self.metric = metric
       self.model_umap = self.load_model()

    def load_model(self):
        model = umap.UMAP(
            n_neighbors = self.n_neighbors,
            n_components = self.n_components,
            metric = self.metric
        )
        return model
    
    def reduce_dimensions(self):
        if self.column not in self.dataset or self.column_index not in self.dataset:
            raise KeyError(f"Columnas '{self.column}' o '{self.column_index}' no encontradas en el dataset.")

        if isinstance(self.dataset, pd.DataFrame):
            vectors = self.dataset[self.column].tolist()
            indexes = self.dataset[self.column_index].astype(str).tolist()
        elif isinstance(self.dataset, dict):
            vectors = [str(t) for t in self.dataset[self.column]]
            indexes = [str(i) for i in self.dataset[self.column_index]]

        reduced_dimensions = self.model_umap.fit_transform(vectors)

        return pd.DataFrame({
            'id': indexes,
            'dimensions' : list(reduced_dimensions)
        })
