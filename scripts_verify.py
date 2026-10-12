from pinecone import Pinecone
from app import config
from langchain_pinecone import PineconeVectorStore
from app.ingest import get_embeddings

pc = Pinecone(api_key=config.PINECONE_API_KEY)
index = pc.Index(config.PINECONE_INDEX_NAME)

stats = index.describe_index_stats()
for ns, info in stats.namespaces.items():
    print(f"{ns}: {info.vector_count} vectors")

store = PineconeVectorStore(
    index_name=config.PINECONE_INDEX_NAME,
    embedding=get_embeddings(),
    namespace="tenant_a",
)
for doc in store.similarity_search("test question about your document", k=2):
    print(doc.metadata, "\n", doc.page_content[:150], "\n")