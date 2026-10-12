import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-saas")

EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIM = 768          # see note below
LLM_MODEL = "gemini-3.5-flash-lite"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBED_BATCH_SIZE = 50        # chunks per embedding request
TOP_K = 4
CHAIN_VERSION = "v3"
JUDGE_MODEL = LLM_MODEL          # you can point this at a different model later
EVAL_DATASET = "rag-saas-tenant-a-golden"