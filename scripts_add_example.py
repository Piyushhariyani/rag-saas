from dotenv import load_dotenv
load_dotenv(override=True)

import sys
from langsmith import Client

# usage: python scripts_add_example.py "question" "corrected answer" PAGE
question, answer, page = sys.argv[1], sys.argv[2], int(sys.argv[3])

Client().create_example(
    inputs={"question": question},
    outputs={"answer": answer, "page": page},
    dataset_name="rag-saas-tenant-a-golden",
)
print("Added example to the golden dataset.")