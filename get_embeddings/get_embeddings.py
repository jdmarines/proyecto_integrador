import tensorflow as tf
import numpy as np
import tensorflow_hub as hub


class GenerateEmbeddings:
    def __init__(self,
                dataset,
                columns,
                model_url = 'https://tfhub.dev/google/universal-sentence-encoder/4'):
        
        self.model_url = model_url
        self.dataset = dataset
        self.columns = columns
        self.model = self.load_model()
        self.embeddings = self.generate_embeddings()
        

    def load_model(self):
        embeddings_model = hub.load(self.model_url)
        return embeddings_model
        
    def generate_embeddings(self):
        generated_embeddings = {}
        for column in self.columns:
            generated_embeddings[r'embeddings_{}'.format(column)] = self.model(self.dataset[column]) 
        return generated_embeddings