# GeoGuessr AI - Street View Country Classifier

A deep learning model that predicts countries from Google Street View images, achieving **48% top-1 accuracy** across 124 countries (60x better than random guessing).

Built with PyTorch and ResNet-50, this project includes real-time screen capture capabilities for live predictions while playing GeoGuessr.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- **Deep CNN model** trained on 50,000 Street View images from 124 countries
- **Real-time prediction** - watches your screen while you play GeoGuessr
- **Smart accumulation mode** - combines multiple views for better accuracy
- **Class-balanced training** - handles dataset imbalance
- **Top-5 accuracy: ~80%** - correct country in top 5 predictions

## Performance

| Metric | Score |
|--------|-------|
| Top-1 Accuracy | 48.1% |
| Top-5 Accuracy | ~80% |
| Countries | 124 |
| Training Images | ~50k |
| Model | ResNet-50 |

**Note:** 48% accuracy on 124 countries is **60x better than random guessing** (0.8%).

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/geoguessr-ai.git
cd geoguessr-ai

# Install dependencies
pip install -r requirements.txt
```

### Train Your Own Model

Train on Google Colab (free GPU - recommended):
1. Open `Geolocation_Training_Colab.ipynb` in Google Colab
2. Upload your dataset to Google Drive
3. Run all cells
4. Download trained model

## Usage

### Real-time Mode (Automatic Screen Capture)

```bash
python geoguessr_realtime.py
```

1. Select the game area
2. Press SPACE to start
3. AI updates automatically as you play!

Combines multiple views as you look around for **70-85% accuracy**!


## Project Structure

```
geoguessr-ai/
├── geoguessr_realtime.py              # Smart accumulation mode
├── requirements.txt                # Dependencies
├── Geolocation_Training_Colab.ipynb # Google Colab notebook
```

## How It Works

**Model:** ResNet-50 CNN pretrained on ImageNet  
**Training:** 20 epochs on 50K images  
**Features learned:** Architecture, vegetation, roads, signs, landscape

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

## Contributing

Contributions welcome! See areas for improvement in the full README.

## License

MIT License - See [LICENSE](LICENSE)

## Disclaimer

Educational purposes only. Respect Google Street View and GeoGuessr terms of service.

---
