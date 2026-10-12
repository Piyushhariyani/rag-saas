import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-saas")

EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIM = 768
LLM_MODEL = "gemini-3.5-flash-lite"

# Ingestion
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBED_BATCH_SIZE = 50

# Retrieval
TOP_K = 4                          # chunks sent to the LLM
FETCH_K = 12                       # candidates fetched before reranking
RERANK_MODEL = "bge-reranker-v2-m3"
USE_RERANK = True                 # False for the baseline run, True for the rerank run

# Versioning and evaluation
CHAIN_VERSION = "v4-rerank"      # change to "v4-rerank" when USE_RERANK = True
JUDGE_MODEL = LLM_MODEL
EVAL_DATASET = "rag-saas-tenant-a-golden"