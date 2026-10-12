from dotenv import load_dotenv
load_dotenv(override=True)   # override=True beats any stale key in your shell

from langsmith import Client

client = Client()
NAME = "rag-saas-tenant-a-golden"

# (question, reference_answer, expected_page)
# (question, reference_answer, expected_page)
EXAMPLES = [
    ("What is the name of the example device in the manual?",
     "The AtlasBook 14, a fictional laptop used only for this sample.", 2),
    ("What sections does the manual include?",
     "Device overview, safety and handling, first-time setup, everyday operation, "
     "troubleshooting, maintenance and data protection, and a support and warranty checklist.", 1),
    ("How should a cleaning solution be applied to the device?",
     "Apply it to a soft, lint-free cloth rather than spraying it directly onto the device, "
     "and never let moisture enter ports or openings.", 2),
    ("What happens after you press the power button during first-time setup?",
     "Wait for the startup screen, then follow the operating system setup flow to choose "
     "language, region, keyboard layout, and network.", 2),
    ("What is the downside of using sleep mode?",
     "Sleep mode may continue to consume power, so the manual recommends shutting down "
     "before transporting the device for an extended period.", 2),
    ("What should I check first if the device does not power on?",
     "Check the wall outlet and the power adapter connection.", 3),
    ("What information should I record before contacting support?",
     "The exact model identifier, operating system version, a description of the issue, "
     "the steps that reproduce it, and any error message.", 3),
    ("What should I do before selling or recycling the device?",
     "Back up your files and follow the operating system's official secure-reset procedure.", 3),
    ("Does the manual establish warranty coverage for the device?",
     "No. The fictional manual cannot establish warranty coverage or provide "
     "model-specific repair instructions.", 4),
    ("What is the capital of France?",
     "I don't know based on the available documents.", None),
]

if not client.has_dataset(dataset_name=NAME):
    ds = client.create_dataset(NAME, description="Golden Q&A for tenant_a")
else:
    ds = client.read_dataset(dataset_name=NAME)

client.create_examples(
    dataset_id=ds.id,
    inputs=[{"question": q} for q, _, _ in EXAMPLES],
    outputs=[{"answer": a, "page": p} for _, a, p in EXAMPLES],
)
print(f"Dataset '{NAME}' ready with {len(EXAMPLES)} examples.")