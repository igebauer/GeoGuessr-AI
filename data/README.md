# data/

This folder stores local data and model files required to run the project.  
Large files are intentionally excluded from GitHub.

## Required files (not included in repo)

Place the following files into this folder before running the real-time model:

- `best_model.pth` — trained PyTorch model weights 
- Dataset zip or extracted dataset (optional, for retraining)

These files are ignored by `.gitignore` to keep the repository lightweight.

## How to obtain them

### Option 1 — Train the model yourself (recommended)
1. Open the training notebook:
- notebooks/Geolocation_Training_Colab.ipynb
  
2. Run all cells in Google Colab with GPU enabled.

3. After training finishes, download:
- `best_model.pth`
- `label_mapping.json`

4. Place both files into this `data/` folder.

### Option 2 — Use your own trained weights
If you already trained a model, simply copy:
- `best_model.pth`
- `label_mapping.json`
into this folder.