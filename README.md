# GeoGuessr-AI

A deep learning project that predicts geographic location from street-level imagery using convolutional neural networks.

This project simulates a real-world computer vision + inference pipeline similar to systems used in:
- autonomous navigation
- satellite imagery analysis
- quantitative alternative data research

Built as an end-to-end ML system, including:
- dataset preprocessing
- model training
- real-time inference
- deployment-ready prediction pipeline

---

## Why This Project Matters

Image-based geolocation is a challenging high-dimensional prediction problem requiring:

- large-scale data handling
- feature extraction from noisy visual inputs
- probabilistic classification
- real-time inference optimization

This project demonstrates the ability to design and implement a full ML
pipeline from raw data to a deployable inference system.

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
```

3. Place model & labels (local or download):
- Place dataset in data/ folder and unzip (download instructions at the bottom)
- Local: put best_model.pth and label_mapping.json into data/ (create the folder)

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

## Training the Model

The model is trained using the provided Jupyter notebook:

Geolocation_Training_Colab.ipynb

This notebook:
- Downloads and preprocesses the dataset
- Creates country labels
- Trains a ResNet50-based classifier in PyTorch
- Saves trained weights and label mappings

### How to train yourself

1. Open the notebook in Google Colab:
   - Upload Geolocation_Training_Colab.ipynb
   - Enable GPU (Runtime → Change runtime → GPU)
2. Download dataset from Kaggle (instructions at bottom)
3. Update dataset paths in notebook if needed
4. Run all cells

After training completes, download:
- best_model.pth
- label_mapping.json

Place them into:
data/

## Model & Data

This repository intentionally does not include model weights or datasets.

Place these files in data/:
- dataset zip file
- best_model.pth — PyTorch model checkpoint
- label_mapping.json — mapping from class index to country name

## Project Structure

```
GeoGuessr-AI/
├── geoguessr_realtime.py     # Real-time overlay + app (refactored)
├── data/                     # gitignored — local models / data go here
├── notebooks/                # Clean notebooks (outputs stripped)
|  └── Geolocation_Training_Colab
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
- Larger dataset scaling

## Contributing & License

Open-source MIT-style license (see LICENSE). Pull requests welcome - please keep large data/model files out of the repo.

## Dataset

This project uses the **GeoGuessr Image Dataset (50k images)** from Kaggle:

https://www.kaggle.com/datasets/ubitquitin/geolocation-geoguessr-images-50k

The dataset is not included in this repository due to size and licensing.

To download:
1. Create a free Kaggle account
2. Install Kaggle CLI:
   pip install kaggle
3. Run:
   kaggle datasets download -d ubitquitin/geolocation-geoguessr-images-50k
4. Unzip into:
   data/

Dataset credit goes to the original Kaggle uploader.
Used for educational and research purposes only.

## Disclaimer

Educational purposes only. Respect Google Street View and GeoGuessr terms of service.
