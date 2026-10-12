from dotenv import load_dotenv
load_dotenv(override=True)

import time

from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import Client, evaluate

from app import config as settings
from app.chain import chain

TENANT = "tenant_a"
client = Client()
judge = ChatGoogleGenerativeAI(model=settings.JUDGE_MODEL, temperature=0)


def target(inputs: dict) -> dict:
    time.sleep(2)  # pace requests for rate limits
    return chain.invoke(
        {"question": inputs["question"]},
        config={"configurable": {"tenant_id": TENANT}},
    )


def retrieval_hit(outputs: dict, reference_outputs: dict) -> dict:
    expected = reference_outputs.get("page")
    if expected is None:  # off-topic question: nothing to retrieve
        return {"key": "retrieval_hit", "score": 1}
    pages = [int(s["page"]) for s in outputs["sources"] if s["page"] is not None]
    return {"key": "retrieval_hit", "score": 1 if int(expected) in pages else 0}


def _grade(prompt: str) -> int:
    time.sleep(2)
    reply = judge.invoke(prompt).content
    if isinstance(reply, list):  # some Gemini models return content blocks
        reply = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in reply
        )
    return 1 if reply.strip().upper().startswith("YES") else 0


def faithfulness(inputs: dict, outputs: dict) -> dict:
    prompt = (
        "You are grading a RAG answer. Reply with only YES or NO.\n"
        f"Question: {inputs['question']}\n"
        f"Answer: {outputs['answer']}\n"
        f"Cited sources: {outputs['sources']}\n"
        "Does the answer cite sources and avoid claims that look invented? "
        "(If the answer is \"I don't know\", reply YES.)"
    )
    return {"key": "faithfulness", "score": _grade(prompt)}


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    prompt = (
        "Reply with only YES or NO.\n"
        f"Question: {inputs['question']}\n"
        f"Reference answer: {reference_outputs['answer']}\n"
        f"Model answer: {outputs['answer']}\n"
        "Does the model answer convey the same facts as the reference answer?"
    )
    return {"key": "correctness", "score": _grade(prompt)}


if __name__ == "__main__":
    examples = list(client.list_examples(dataset_name=settings.EVAL_DATASET))
    print("Examples loaded:", len(examples))
    if not examples:
        raise SystemExit(
            f"No examples found in dataset {settings.EVAL_DATASET!r}. "
            "Check EVAL_DATASET in app/config.py."
        )

    results = evaluate(
        target,
        data=examples,
        evaluators=[retrieval_hit, faithfulness, correctness],
        experiment_prefix=f"{settings.CHAIN_VERSION}-k{settings.TOP_K}",
        max_concurrency=1,
        client=client,
        metadata={
            "chain_version": settings.CHAIN_VERSION,
            "top_k": settings.TOP_K,
            "fetch_k": settings.FETCH_K,
            "rerank": settings.USE_RERANK,
            "rerank_model": settings.RERANK_MODEL,
            "chunk_size": settings.CHUNK_SIZE,
            "llm_model": settings.LLM_MODEL,
        },
    )
    print("Experiment finished. Open LangSmith, Datasets & Experiments.")