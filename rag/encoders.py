from langchain_huggingface import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder
from core.config import settings

def get_encoders() -> tuple[HuggingFaceEmbeddings, BM25Encoder]:
    print("[RAG] Loading encoders...")
    dense  = HuggingFaceEmbeddings(model_name=settings.DENSE_ENCODER)
    sparse = BM25Encoder()
    sparse.load(settings.SPARSE_ENCODER)
    return dense, sparse
