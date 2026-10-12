from dotenv import load_dotenv
load_dotenv(override=True)

from langsmith import Client
from app import config

HARD = [
    ("Which safety rules apply when cleaning and charging the device?",
     "Disconnect power and unplug cables before cleaning, use a soft lint-free cloth with any "
     "solution applied to the cloth, and use only a charger that meets the manufacturer's "
     "voltage, current, connector, and safety requirements.", 2),
    ("What should I avoid doing on public Wi-Fi?",
     "Avoid transmitting sensitive information over unknown networks, and prefer encrypted websites.", 3),
    ("Can I open the battery enclosure to repair a swollen battery?",
     "No. Do not open the battery enclosure or attempt to repair swollen, leaking, or damaged batteries.", 2),
    ("What should I do if the device is unusually slow?",
     "Save work and restart, check storage and close unneeded applications, install updates and "
     "run a security scan, and review recent software changes using official recovery guidance.", 3),
    ("Who manufactures the AtlasBook 14?",
     "I don't know based on the available documents.", None),
]

client = Client()
client.create_examples(
    dataset_name=config.EVAL_DATASET,
    inputs=[{"question": q} for q, _, _ in HARD],
    outputs=[{"answer": a, "page": p} for _, a, p in HARD],
)
print(f"Added {len(HARD)} examples.")