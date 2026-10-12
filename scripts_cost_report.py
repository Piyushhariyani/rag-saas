from collections import defaultdict
from datetime import datetime, timedelta, timezone

from langsmith import Client

from app import config

client = Client()
since = datetime.now(timezone.utc) - timedelta(days=7)

stats = defaultdict(lambda: {"runs": 0, "errors": 0, "tokens": 0, "cost": 0.0, "latency": []})

for run in client.list_runs(
    project_name=config_project if (config_project := __import__("os").getenv("LANGSMITH_PROJECT")) else None,
    is_root=True,
    start_time=since,
):
    meta = (run.extra or {}).get("metadata", {})
    tenant = meta.get("tenant_id", "unknown")
    s = stats[tenant]
    s["runs"] += 1
    s["errors"] += 1 if run.error else 0
    s["tokens"] += run.total_tokens or 0
    s["cost"] += float(run.total_cost or 0)
    if run.start_time and run.end_time:
        s["latency"].append((run.end_time - run.start_time).total_seconds())

print(f"{'tenant':<10} {'runs':>5} {'errors':>6} {'tokens':>8} {'cost($)':>9} {'avg_s':>6} {'max_s':>6}")
for tenant, s in sorted(stats.items()):
    lat = s["latency"]
    avg = sum(lat) / len(lat) if lat else 0
    print(f"{tenant:<10} {s['runs']:>5} {s['errors']:>6} {s['tokens']:>8} "
          f"{s['cost']:>9.4f} {avg:>6.1f} {max(lat, default=0):>6.1f}")