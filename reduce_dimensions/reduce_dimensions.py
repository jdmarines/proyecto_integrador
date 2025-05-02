import umap
import numpy as np
import tensorflow as tf
import pandas as pd


class UmapReduction:
    def __init__(self, columns, dataset, n_neighbors = 15, n_components = 2, metric = 'cosine'):
       self.dataset = dataset
       self.columns = columns
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

    def unbatch_data(self):
        unbatched_data = {}
        for column_name in self.columns:
            unbatched_data[column_name] = []
            for batch in self.dataset:
                unbatched_data[column_name].append(batch[column_name].numpy())
         
        for column in self.columns:
            unbatched_data[column] = np.concatenate(unbatched_data[column], axis = 0)

        return  unbatched_data
    
    def reduce_dimensions(self):
        flat_data = self.unbatch_data()
        reduced_dimensions = {}
        for column_name in flat_data.items():
            reduced_dimensions[column_name] = self.model.fit_transform(flat_data[column_name])
        return reduced_dimensions