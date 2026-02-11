# GeoGuessr-AI

**GeoGuessr-AI** — a research/demo project that uses image-based deep learning to predict the country displayed in Google Street View-style images and *accumulates multiple views* to form a stronger guess. Built with PyTorch and a lightweight GUI overlay for real-time inference while you move in the game.

---

## Key highlights
- **Task:** Predict country from street-view images using a CNN backbone (ResNet-50).
- **Method:** Aggregates predictions from multiple frames using weighted averaging, max-vote, and consensus strategies to increase robustness.
- **Realtime demo:** Tkinter overlay captures game window and shows current + accumulated guesses with confidence.
- **Engineering:** Clean, modular Python with a downloader helper to keep large files out of Git history.

---

## Performance

| Metric | Score |
|--------|-------|
| Top-1 Accuracy | 48.1% |
| Top-5 Accuracy | ~80% |
| Countries | 124 |
| Training Images | ~50k |
| Model | ResNet-50 |

## Quick start

1. Clone:
```bash
git clone https://github.com/igebauer/GeoGuessr-AI.git
cd GeoGuessr-AI
```

2. Install dependencies
```bash
pip install -r requirements.txt
# If you plan to use the provided downloader:
pip install gdown
```

3. Place model & labels (local or download):
- Local: put best_model.pth and label_mapping.json into data/ (create the folder).
- Or use the helper script:
```bash
python scripts/download_weights.py
```

4. Run the real-time overlay:
```bash
python geoguessr_realtime.py --model data/best_model.pth --labels data/label_mapping.json
```
Controls (overlay window):
- SPACE — Start/Stop predictions
- R — Reset accumulation (new location)
- M — Cycle aggregation method (Weighted Avg / Max Vote / Consensus)
- + / - — Adjust update interval
- ESC — Quit

## Model & Data

This repository intentionally does not include model weights or datasets.

Recommended hosting options:
- Hugging Face Hub for models (recommended)
- Google Drive / S3 for datasets (use scripts/download_weights.py)

Place these files in data/:
- best_model.pth — PyTorch model checkpoint
- label_mapping.json — mapping from class index to country name

## Project Structure

```
GeoGuessr-AI/
├── geoguessr_realtime.py     # Real-time overlay + app (refactored)
├── scripts/
│   └── download_weights.py   # Helper downloader for large assets
├── data/                     # gitignored — local models / data go here
├── notebooks/                # Clean notebooks (outputs stripped)
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
└── .gitattributes
```

## How it works (short)

1. Model predicts a probability distribution over classes for each captured frame.
2. PredictionAccumulator stores recent predictions (configurable window) and computes an aggregated ranking using the selected method.
3. Overlay GUI (Tkinter) displays both current-view probabilities and accumulated decisions with confidence.

## Example Results

```
Analyzing: street_view.jpg

1. France              67.2%  ████████████████████
2. Belgium             12.5%  ████
3. Switzerland          8.3%  ██
4. Germany              5.7%  █
5. Luxembourg           3.2%  

Best guess: France
```

## Ideas to improve / metrics to add

- Report Top-1 / Top-5 accuracy on validation set and median error distance (km).
- Compare backbone choices (ResNet-50 vs. CLIP / ViT).
- Add map visualization (lat/long predictions) and a small technical appendix showing feature engineering.
- Add unit tests and a small benchmark for inference speed (FPS).

## Contributing & License

Open-source MIT-style license (see LICENSE). Pull requests welcome - please keep large data/model files out of the repo.

## Disclaimer

Educational purposes only. Respect Google Street View and GeoGuessr terms of service.
