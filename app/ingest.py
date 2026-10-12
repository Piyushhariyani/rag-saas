import hashlib
import time
from datetime import date
from pathlib import Path

from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from app import config

SUPPORTED = {".pdf", ".txt", ".md"}


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        output_dimensionality=config.EMBEDDING_DIM,
    )


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_file(path: Path):
    if path.suffix.lower() == ".pdf":
        return PyPDFLoader(str(path)).load()
    return TextLoader(str(path), encoding="utf-8").load()


def already_ingested(index, tenant_id: str, first_id: str) -> bool:
    res = index.fetch(ids=[first_id], namespace=tenant_id)
    return first_id in res.vectors


def ingest_file(path: Path, tenant_id: str) -> int:
    """Ingest one file into the tenant's namespace. Returns chunks added."""
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = pc.Index(config.PINECONE_INDEX_NAME)

    fhash = file_hash(path)
    if already_ingested(index, tenant_id, f"{fhash}-0"):
        print(f"  skip (already ingested): {path.name}")
        return 0

    pages = load_file(path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)

    ids = []
    for i, chunk in enumerate(chunks):
        page = chunk.metadata.get("page")
        chunk.metadata = {
            "tenant_id": tenant_id,
            "source": path.name,
            "page": (page + 1) if isinstance(page, int) else 1,
            "file_hash": fhash,
            "upload_date": date.today().isoformat(),
            "chunk_index": i,
        }
        ids.append(f"{fhash}-{i}")

    store = PineconeVectorStore(
        index_name=config.PINECONE_INDEX_NAME,
        embedding=get_embeddings(),
        namespace=tenant_id,
    )

    bs = config.EMBED_BATCH_SIZE
    for start in range(0, len(chunks), bs):
        batch_docs = chunks[start:start + bs]
        batch_ids = ids[start:start + bs]
        for attempt in range(5):
            try:
                store.add_documents(batch_docs, ids=batch_ids)
                break
            except Exception as e:           # rate limits (429) on free tier
                wait = 2 ** attempt * 5
                print(f"  retry in {wait}s: {type(e).__name__}")
                time.sleep(wait)
        else:
            raise RuntimeError(f"Failed to ingest batch starting at {start}")
        time.sleep(1)                        # gentle pacing

    print(f"  ingested {len(chunks)} chunks: {path.name}")
    return len(chunks)


def ingest_tenant_folder(folder: Path) -> int:
    tenant_id = folder.name                  # folder name = tenant ID
    total = 0
    print(f"Tenant: {tenant_id}")
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in SUPPORTED:
            total += ingest_file(path, tenant_id)
    return total