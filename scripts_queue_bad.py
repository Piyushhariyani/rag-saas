from dotenv import load_dotenv
load_dotenv(override=True)

import os
from langsmith import Client

client = Client()
PROJECT = os.getenv("LANGSMITH_PROJECT", "rag-saas-dev")
QUEUE = "rag-saas-review"

runs = list(client.list_runs(
    project_name=PROJECT,
    is_root=True,
    filter='and(eq(feedback_key, "user_score"), eq(feedback_score, 0))',
))
print("Thumbs-down runs found:", len(runs))
if not runs:
    raise SystemExit("Nothing to queue yet.")

queues = list(client.list_annotation_queues(name=QUEUE))
queue = queues[0] if queues else client.create_annotation_queue(
    name=QUEUE, description="Answers users rated thumbs-down"
)

client.add_runs_to_annotation_queue(queue.id, run_ids=[r.id for r in runs])
print(f"Added {len(runs)} runs to queue '{QUEUE}'.")