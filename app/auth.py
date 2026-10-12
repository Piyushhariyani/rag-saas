import os
import hmac
from fastapi import HTTPException
# pyrefly: ignore [missing-import]
from app import config

def _load_keys() -> dict:
    mapping = {}
    for tenant in ("tenant_a", "tenant_b", "tenant_c"):
        key = os.getenv(f"KEY_{tenant.upper()}")
        if key:
            mapping[key] = tenant
    return mapping


API_KEYS = _load_keys()


def tenant_from_key(api_key: str | None) -> str:
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")
    for known_key, tenant in API_KEYS.items():
        if hmac.compare_digest(known_key, api_key):
            return tenant
    raise HTTPException(status_code=401, detail="Invalid API key")