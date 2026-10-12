from pinecone import Pinecone, ServerlessSpec
from app import config

pc = Pinecone(api_key=config.PINECONE_API_KEY)

if config.PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=config.PINECONE_INDEX_NAME,
        dimension=config.EMBEDDING_DIM,   # 768, must match the embedding model
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("Index created.")
else:
    print("Index already exists.")

print(pc.describe_index(config.PINECONE_INDEX_NAME))