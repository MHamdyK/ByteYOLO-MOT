# src/utils/io.py
import os
import requests
from pathlib import Path

ASSETS = {
    "best.pt": "https://github.com/your-org/byteyolo-mot/releases/download/v0.1.0/best.pt",
    "demo.gif": "https://github.com/your-org/byteyolo-mot/releases/download/v0.1.0/demo.gif",
}

def fetch_assets(dest_weights="models/weights", dest_demo="docs/demo"):
    os.makedirs(dest_weights, exist_ok=True)
    os.makedirs(dest_demo, exist_ok=True)
    for name, url in ASSETS.items():
        out_path = Path(dest_weights if name.endswith(".pt") else dest_demo) / name
        if not out_path.exists():
            print(f"Downloading {name}...")
            r = requests.get(url, stream=True)
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
            print(f"Saved → {out_path}")
        else:
            print(f"{name} already exists, skipping.")
