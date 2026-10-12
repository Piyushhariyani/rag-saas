import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

BASE = "http://localhost:8000"
TENANTS = ["tenant_a", "tenant_b", "tenant_c"]

QUESTIONS = [
    "What is this document about?",
    "Summarize the main sections.",
    "What safety warnings are mentioned?",
    "What are the key rules or conditions described?",
    "Who is this document intended for?",
    "What is the capital of France?",          # off-topic, should say "I don't know"
]

for tenant in TENANTS:
    key = os.getenv(f"KEY_{tenant.upper()}")
    for q in QUESTIONS:
        r = requests.post(
            f"{BASE}/query/invoke",
            headers={"x-api-key": key},
            json={"input": {"question": q}},
            timeout=120,
        )
        print(tenant, r.status_code, q)
        time.sleep(3)    # stay under the Gemini free-tier rate limit