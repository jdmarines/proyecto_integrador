from embeddings.storage.base import EmbeddingStorageStrategy
from embeddings.decoders.base import EmbeddingGenerationStrategy

class GenerateEmbeddings:
    def __init__(self,
                save_strategy: EmbeddingStorageStrategy,
                embedding_strategy: EmbeddingGenerationStrategy):
        self.embedding_strategy = embedding_strategy
        self.save_strategy = save_strategy

    def generate_embeddings(self):
        return self.embedding_strategy.generate_embeddings()

    def save_embeddings(self, embeddings):
        self.save_strategy.save(embeddings)




