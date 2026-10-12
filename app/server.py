from dotenv import load_dotenv
load_dotenv(override=True)

import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from langserve import add_routes
from langsmith import Client
from pydantic import BaseModel, Field

from app import config as settings
from app.auth import tenant_from_key
from app.chain import chain
from app.ingest import ingest_file

app = FastAPI(title="RAG SaaS", version="0.2.0")
ls_client = Client()

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


class FeedbackIn(BaseModel):
    run_id: str
    score: int = Field(ge=0, le=1)  # 1 = thumbs up, 0 = thumbs down
    comment: str | None = Field(default=None, max_length=1000)


async def per_req_config_modifier(config: dict, request: Request) -> dict:
    tenant_id = tenant_from_key(request.headers.get("x-api-key"))

    configurable = dict(config.get("configurable") or {})
    configurable["tenant_id"] = tenant_id  # overwrites any client-supplied value
    config["configurable"] = configurable

    config.setdefault("metadata", {}).update({
        "tenant_id": tenant_id,
        "chain_version": settings.CHAIN_VERSION,
        "llm_model": settings.LLM_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "top_k": settings.TOP_K,
    })
    config["tags"] = list(config.get("tags") or []) + [f"tenant:{tenant_id}"]
    return config


add_routes(
    app,
    chain,
    path="/query",
    per_req_config_modifier=per_req_config_modifier,
)


@app.post("/ingest")
async def ingest(file: UploadFile = File(...), x_api_key: str | None = Header(default=None)):
    tenant_id = tenant_from_key(x_api_key)

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / Path(file.filename).name  # strips any path parts
        size = 0
        with dest.open("wb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="File too large (20 MB max)")
                f.write(chunk)
        chunks = ingest_file(dest, tenant_id)

    return {"tenant": tenant_id, "file": file.filename, "chunks_added": chunks}


@app.post("/feedback")
def feedback(body: FeedbackIn, x_api_key: str | None = Header(default=None)):
    tenant_id = tenant_from_key(x_api_key)

    # Traces upload in the background, so retry briefly before giving up.
    run = None
    for _ in range(5):
        try:
            run = ls_client.read_run(body.run_id)
            break
        except Exception as e:
            print("read_run failed:", repr(e))  # visible in the server terminal
            time.sleep(2)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    # Security: a tenant may only rate its own runs.
    owner = (run.extra or {}).get("metadata", {}).get("tenant_id")
    if owner != tenant_id:
        print("owner:", owner, "caller:", tenant_id)
        raise HTTPException(status_code=404, detail="Run not found")  # same error, so run IDs can't be probed

    ls_client.create_feedback(
        run_id=body.run_id,
        key="user_score",
        score=body.score,
        comment=body.comment,
    )
    return {"status": "recorded"}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=True)