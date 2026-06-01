from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone, ServerlessSpec
from core.config import settings

def setup_hybrid_retriever(dense_encoder, sparse_encoder, k=None):
    pc    = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(settings.PINECONE_INDEX_NAME)
    return PineconeHybridSearchRetriever(
        embeddings=dense_encoder, sparse_encoder=sparse_encoder,
        index=index, top_k=k or settings.RAG_TOP_K, alpha=0.5)

def create_pinecone_index():
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    if settings.PINECONE_INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
        pc.create_index(
            name=settings.PINECONE_INDEX_NAME,
            dimension=settings.PINECONE_DIMENSION,
            metric=settings.PINECONE_METRIC,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        print(f"[RAG] Index '{settings.PINECONE_INDEX_NAME}' created.")

def prepare_documents(data_dicts: list[dict], sparse_encoder) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""])
    texts = [f"Context: {d['Context']}\nResponse: {d['Response']}" for d in data_dicts]
    sparse_encoder.fit(texts)
    docs = []
    for d in data_dicts:
        for i, chunk in enumerate(splitter.split_text(str(d["Response"]))):
            docs.append(Document(
                page_content=chunk,
                metadata={"Context": str(d["Context"]), "Response": chunk, "Chunk_Part": i + 1}))
    return docs

def add_documents(retriever, documents: list[Document]) -> None:
    retriever.add_texts(
        texts=[d.page_content for d in documents],
        metadatas=[d.metadata for d in documents])
    print(f"[RAG] Added {len(documents)} documents to Pinecone.")
    