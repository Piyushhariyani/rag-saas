import sys
# pyrefly: ignore [missing-import]
from app.chain import chain

tenant = sys.argv[1]
question = " ".join(sys.argv[2:])

result = chain.invoke(
    {"question": question},
    config={"configurable": {"tenant_id": tenant}},
)
print("\nANSWER:\n", result["answer"])
print("\nSOURCES:", result["sources"])