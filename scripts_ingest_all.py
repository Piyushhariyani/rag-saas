from pathlib import Path
from app.ingest import ingest_tenant_folder

DATA_DIR = Path("data")

for folder in sorted(DATA_DIR.iterdir()):
    if folder.is_dir() and folder.name.startswith("tenant_"):
        ingest_tenant_folder(folder)

print("Done.")