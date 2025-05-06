import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import pandas as pd
from ..base import EmbeddingGenerationStrategy

class GenerateEmbeddingsTF(EmbeddingGenerationStrategy):
    def __init__(self,
                dataset,
                column,
                column_index,
                model_url = 'https://tfhub.dev/google/universal-sentence-encoder/4'):
        
        self.model_url = model_url
        self.column_index = column_index
        self.dataset = dataset
        self.column = column
        self.model = self.load_model()
        
    def load_model(self):
        return hub.load(self.model_url)
    
    def generate_embeddings(self):
        if self.column not in self.dataset or self.column_index not in self.dataset:
            raise KeyError(f"Columnas '{self.column}' o '{self.column_index}' no encontradas en el dataset.")

        if isinstance(self.dataset, pd.DataFrame):
            texts = self.dataset[self.column].astype(str).tolist()
            indexes = self.dataset[self.column_index].astype(str).tolist()
        elif isinstance(self.dataset, dict):
            texts = [str(t) for t in self.dataset[self.column]]
            indexes = [str(i) for i in self.dataset[self.column_index]]
        else:
            raise ValueError("Formato de Dataset no soportado")
        embeddings = self.model(texts).numpy()
        return pd.DataFrame({
            'id': indexes,
            "embeddings": list(embeddings)
        })