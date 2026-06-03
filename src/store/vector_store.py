from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams
from store import Embeddings


class QdrantStore:
    def __init__(self):
        self.embeddings = Embeddings().get_model()
        self.client = QdrantClient(
            path="/tmp/langchain_qdrant3", force_disable_check_same_thread=True
        )
        self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        self.qdrant = QdrantVectorStore(
            client=self.client,
            collection_name="NLP_Project",
            embedding=self.embeddings,
            sparse_embedding=self.sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
            validate_collection_config=False,
        )

    def return_qdrant(self):
        return self.qdrant

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass


if __name__ == "__main__":
    store = QdrantStore()
    print(
        store.qdrant.search(
            "What is the best way to deal with anxiety?", k=3, search_type="similarity"
        )
    )
    store.client.close()
