import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import math
import os

class TopicModeler:
    def __init__(self, csv_path: str, text_column: str = 'text', model_path: str = 'bertopic_model'):
        self.csv_path = csv_path
        self.text_column = text_column
        self.model_path = model_path
        self.df = None
        self.model = None
        self.topics = None
        self.probs = None

    def load_data(self):
        self.df = pd.read_csv(self.csv_path)
        if self.text_column not in self.df.columns:
            raise ValueError(f"La columna '{self.text_column}' no está en el archivo CSV.")
        self.df[self.text_column] = self.df[self.text_column].astype(str)
        print(f"Datos cargados. Total de registros: {len(self.df)}")

    def preprocess_text(self):
        self.df[self.text_column] = self.df[self.text_column].str.lower().str.strip()

    def fit_model(self):
        if os.path.exists(f"{self.model_path}.bertopic"):
            print(f"Modelo existente encontrado en {self.model_path}.bertopic. Cargando modelo...")
            self.model = BERTopic.load(f"{self.model_path}.bertopic")
        else:
            print("Entrenando nuevo modelo BERTopic...")

            embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")
            umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')

            # Reducción de clusters pequeños
            cluster_model = HDBSCAN(min_cluster_size=50, min_samples=10, metric='euclidean')

            self.model = BERTopic(
                embedding_model=embedding_model,
                vectorizer_model=vectorizer_model,
                umap_model=umap_model,
                hdbscan_model=cluster_model,
                language="multilingual"
            )

            self.topics, self.probs = self.model.fit_transform(self.df[self.text_column])

            # Reducimos la cantidad total de temas a 20 (ajustable)
            # self.model.reduce_topics(self.df[self.text_column], nr_topics=20)

            print("Modelo entrenado y reducido a 20 temas.")
            self.model.save(f"{self.model_path}.bertopic")
            print(f"Modelo guardado en {self.model_path}.bertopic")

    def show_topics(self, n: int = 10):
        print("\n Top temas:")
        for topic_id, topic in self.model.get_topics().items():
            if topic_id == -1:  # Ignorar outliers
                continue
            print(f"Tema {topic_id}: {[word for word, _ in topic][:n]}")

    def visualize_topics(self):
        self.model.visualize_topics().show()

    def visualize_hierarchy(self):
        self.model.visualize_hierarchy().show()

    def create_subtopics(self):
        print("Generando subtemas jerárquicos...")
        hierarchical_topics = self.model.hierarchical_topics(self.df[self.text_column])
        self.model.visualize_hierarchy(hierarchical_topics=hierarchical_topics).show()



    def save_results(self, output_path: str = "resultados_bertopic.csv"):
        if self.topics is None:
            self.topics, self.probs = self.model.transform(self.df[self.text_column])
        self.df["topic"] = self.topics
        self.df.to_csv(output_path, index=False)
        print(f"Resultados guardados en: {output_path}")