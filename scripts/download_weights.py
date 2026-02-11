# scripts/download_weights.py
"""
Downloader for models / large data. Edit FILE_ID and DEST for each asset.
Usage:
    python scripts/download_weights.py
This keeps large files out of git while making it easy for users to get them.
"""
import os
import sys

try:
    import gdown
except Exception:
    print("Please install gdown: pip install gdown")
    sys.exit(1)

DOWNLOADS = [
    # (google drive file id, target path)
    ("1AE2Y80P3YSb4GMwoPqBBHhE8t1Mrnu_v", "data/archive.zip"),
    # ("<MODEL_FILE_ID>", "data/best_model.pth"),
]

os.makedirs("data", exist_ok=True)

for file_id, out_path in DOWNLOADS:
    if os.path.exists(out_path):
        print(f"Already exists: {out_path}")
        continue
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading to {out_path} ...")
    gdown.download(url, out_path, quiet=False)
    print("Done.")