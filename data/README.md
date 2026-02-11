# data/ (ignored)

This folder is intentionally `.gitignore`d. Place large assets here.

Files:
- `best_model.pth` — your trained PyTorch weights (do NOT commit to GitHub)
- `archive.zip` — original dataset (do NOT commit)

Options for hosting:
- Google Drive (use `scripts/download_weights.py`)
- Hugging Face Hub (recommended for models)
- S3 / private storage

Example:
1. Put `best_model.pth` into `data/`
2. Run: `python geoguessr_realtime.py --model data/best_model.pth --labels data/label_mapping.json`