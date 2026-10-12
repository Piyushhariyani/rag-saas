from dotenv import load_dotenv
load_dotenv(override=True)

import os
import sys
from collections import defaultdict

from langsmith import Client, evaluate

from app import config as settings
from evals.run_evals import correctness, faithfulness, retrieval_hit, target

THRESHOLDS = {
    "correctness": 0.80,
    "faithfulness": 0.90,
    "retrieval_hit": 0.90,
}
LIMIT = int(os.getenv("EVAL_LIMIT", "0")) or None   # cap examples to save quota


def main() -> int:
    client = Client()
    examples = list(client.list_examples(dataset_name=settings.EVAL_DATASET))
    if LIMIT:
        examples = examples[:LIMIT]
    if not examples:
        print("FAIL: no examples found")
        return 1

    results = evaluate(
        target,
        data=examples,
        evaluators=[retrieval_hit, faithfulness, correctness],
        experiment_prefix=f"ci-{settings.CHAIN_VERSION}",
        max_concurrency=1,
        client=client,
        metadata={"chain_version": settings.CHAIN_VERSION, "source": "ci"},
    )

    scores = defaultdict(list)
    for row in results:
        for er in row["evaluation_results"]["results"]:
            if er.score is not None:
                scores[er.key].append(er.score)

    failed = False
    print(f"\nEvaluated {len(examples)} examples")
    for key, minimum in THRESHOLDS.items():
        vals = scores.get(key, [])
        if len(vals) < len(examples):          # missing scores = failed runs
            print(f"FAIL {key}: only {len(vals)}/{len(examples)} scored")
            failed = True
            continue
        avg = sum(vals) / len(vals)
        ok = avg >= minimum
        print(f"{'PASS' if ok else 'FAIL'} {key}: {avg:.2f} (minimum {minimum:.2f})")
        failed = failed or not ok
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())