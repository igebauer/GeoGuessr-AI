# GeoGuessr AI app: Combines predictions to make better guesses

# imports
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image, ImageGrab
import json
import tkinter as tk
from tkinter import ttk
import threading
import time
from pathlib import Path
import sys
from collections import defaultdict, Counter
import numpy as np


class GeoLocalizationModel(nn.Module):
    def __init__(self, num_classes, backbone='resnet50'):
        super().__init__()
        
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=False)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class SmartPredictor:
    def __init__(self, model_path, label_mapping_path):
        with open(label_mapping_path, 'r') as f:
            self.label_mapping = json.load(f)
            self.label_mapping = {int(k): v for k, v in self.label_mapping.items()}
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(model_path, map_location=self.device)
        num_classes = checkpoint['num_classes']
        
        self.model = GeoLocalizationModel(num_classes, backbone='resnet50')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded: {num_classes} countries | Device: {self.device}")
    
    def predict(self, image, top_k=10):
        # Get prediction probabilities for all countries
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
        
        # Return all probabilities as dict
        result = {}
        for idx, prob in enumerate(probabilities.cpu().numpy()):
            country = self.label_mapping[idx]
            result[country] = float(prob)
        
        return result


class PredictionAccumulator:
    # Accumulates predictions over multiple views
    
    def __init__(self):
        self.predictions = []  # List of probability dicts
        self.max_predictions = 20  # Keep last N predictions
    
    def add(self, prediction_probs):
        # Add a new prediction
        self.predictions.append(prediction_probs)
        
        # Keep only recent predictions
        if len(self.predictions) > self.max_predictions:
            self.predictions.pop(0)
    
    def get_accumulated(self, top_k=5, method='weighted_average'):
        
        # Get prediction across all views
        # Methods:
        # - weighted_average: Average probabilities, recent views weighted more
        # - max_vote: Take highest probability seen for each country
        # - consensus: Countries that appear consistently in top predictions
       
        if not self.predictions:
            return []
        
        if method == 'weighted_average':
            # Weight recent predictions more heavily
            weights = np.linspace(0.5, 1.0, len(self.predictions))
            weights = weights / weights.sum()
            
            # Weighted average of probabilities
            accumulated = defaultdict(float)
            for pred, weight in zip(self.predictions, weights):
                for country, prob in pred.items():
                    accumulated[country] += prob * weight
            
        elif method == 'max_vote':
            # Take maximum probability ever seen for each country
            accumulated = defaultdict(float)
            for pred in self.predictions:
                for country, prob in pred.items():
                    accumulated[country] = max(accumulated[country], prob)
        
        elif method == 'consensus':
            # Boost countries that appear consistently in top 5
            accumulated = defaultdict(float)
            for pred in self.predictions:
                # Get top 5 from this prediction
                top_5 = sorted(pred.items(), key=lambda x: x[1], reverse=True)[:5]
                for rank, (country, prob) in enumerate(top_5):
                    # Bonus for appearing in top 5, more for higher ranks
                    bonus = (5 - rank) * 0.2
                    accumulated[country] += prob + bonus
        
        # Sort by accumulated score
        sorted_results = sorted(accumulated.items(), key=lambda x: x[1], reverse=True)
        
        # Normalize to percentages
        total = sum(score for _, score in sorted_results[:top_k])
        if total > 0:
            results = [(country, (score/total) * 100) for country, score in sorted_results[:top_k]]
        else:
            results = [(country, 0.0) for country, _ in sorted_results[:top_k]]
        
        return results
    
    def get_current(self, top_k=5):
        # Get just the most recent prediction
        if not self.predictions:
            return []
        
        latest = self.predictions[-1]
        sorted_results = sorted(latest.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return [(country, prob * 100) for country, prob in sorted_results]
    
    def reset(self):
        # Clear all accumulated predictions (new location)
        self.predictions.clear()
    
    def get_confidence_trend(self):
        # Get confidence trend over time
        if len(self.predictions) < 2:
            return "stable"
        
        # Check if top prediction is becoming more confident
        recent_tops = []
        for pred in self.predictions[-5:]:
            top_country, top_prob = max(pred.items(), key=lambda x: x[1])
            recent_tops.append(top_prob)
        
        if len(recent_tops) >= 3:
            # Check if confidence is increasing
            if recent_tops[-1] > recent_tops[-2] > recent_tops[-3]:
                return "increasing"
            elif recent_tops[-1] < recent_tops[-2] < recent_tops[-3]:
                return "decreasing"
        
        return "stable"
    
    def get_sample_count(self):
        # Number of views accumulated
        return len(self.predictions)


class RegionSelector:
    def __init__(self):
        self.region = None
        self.start_x = None
        self.start_y = None
        
        self.root = tk.Tk()
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-alpha', 0.3)
        self.root.configure(bg='black')
        
        self.canvas = tk.Canvas(self.root, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        instructions = tk.Label(self.root, 
                               text="Click and drag to select the game area\nPress ESC to use full screen",
                               font=('Arial', 20, 'bold'),
                               bg='black', fg='white')
        instructions.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        self.canvas.bind('<Button-1>', self.on_click)
        self.canvas.bind('<B1-Motion>', self.on_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_release)
        self.root.bind('<Escape>', lambda e: self.use_fullscreen())
        
        self.rect = None
    
    def on_click(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.rect:
            self.canvas.delete(self.rect)
    
    def on_drag(self, event):
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline='red', width=3
        )
    
    def on_release(self, event):
        x1, y1 = min(self.start_x, event.x), min(self.start_y, event.y)
        x2, y2 = max(self.start_x, event.x), max(self.start_y, event.y)
        
        if (x2 - x1) > 100 and (y2 - y1) > 100:
            self.region = (x1, y1, x2, y2)
            self.root.quit()
            self.root.destroy()
    
    def use_fullscreen(self):
        self.region = None
        self.root.quit()
        self.root.destroy()
    
    def select(self):
        self.root.mainloop()
        return self.region


class SmartOverlay:
    def __init__(self, predictor, region=None):
        self.predictor = predictor
        self.accumulator = PredictionAccumulator()
        self.region = region
        self.running = False
        self.paused = True
        self.update_interval = 1.5
        self.aggregation_method = 'weighted_average'
        
        # Create window
        self.root = tk.Tk()
        self.root.title("GeoGuessr AI")
        self.root.attributes('-topmost', True)
        
        # Larger window to show both current and accumulated
        window_width = 450
        window_height = 500
        screen_width = self.root.winfo_screenwidth()
        x = screen_width - window_width - 20
        y = 20
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#0a0e27', padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = tk.Frame(main_frame, bg='#0a0e27')
        header_frame.pack(fill=tk.X, pady=(0, 5))
        
        title = tk.Label(header_frame, text="GeoGuessr AI", 
                        font=('Arial', 16, 'bold'), 
                        bg='#0a0e27', fg='#00d4ff')
        title.pack(side=tk.LEFT)
        
        self.status_dot = tk.Label(header_frame, text="⏸", 
                                   font=('Arial', 18),
                                   bg='#0a0e27', fg='#ff6b6b')
        self.status_dot.pack(side=tk.RIGHT)
        
        smart_label = tk.Label(main_frame, text="Accumulating Views", 
                              font=('Arial', 10, 'bold'),
                              bg='#0a0e27', fg='#00ff88')
        smart_label.pack()
        
        # Stats
        self.stats_label = tk.Label(main_frame, text="Views: 0 | Confidence: --",
                                    font=('Arial', 9),
                                    bg='#0a0e27', fg='#95a5a6')
        self.stats_label.pack(pady=(0, 5))
        
        # ACCUMULATED prediction (main)
        acc_title = tk.Label(main_frame, text="ACCUMULATED GUESS (All Views)",
                            font=('Arial', 11, 'bold'),
                            bg='#0a0e27', fg='#ffd700')
        acc_title.pack(pady=(5, 3))
        
        self.accumulated_frame = tk.Frame(main_frame, bg='#0a0e27')
        self.accumulated_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Separator
        tk.Frame(main_frame, height=2, bg='#34495e').pack(fill=tk.X, pady=5)
        
        # CURRENT prediction (secondary)
        curr_title = tk.Label(main_frame, text="Current View",
                             font=('Arial', 9),
                             bg='#0a0e27', fg='#95a5a6')
        curr_title.pack(pady=(5, 3))
        
        self.current_frame = tk.Frame(main_frame, bg='#0a0e27')
        self.current_frame.pack(fill=tk.X)
        
        # Controls
        controls_frame = tk.Frame(main_frame, bg='#1a2332', relief=tk.SOLID, borderwidth=1)
        controls_frame.pack(fill=tk.X, pady=(10, 0))
        
        tk.Label(controls_frame, text="SPACE: Start/Stop | R: Reset | M: Method | ESC: Quit",
                font=('Arial', 7), bg='#1a2332', fg='#95a5a6').pack(pady=3)
        
        # Bind keys
        self.root.bind('<space>', lambda e: self.toggle_pause())
        self.root.bind('<Escape>', lambda e: self.quit())
        self.root.bind('r', lambda e: self.reset_accumulation())
        self.root.bind('R', lambda e: self.reset_accumulation())
        self.root.bind('m', lambda e: self.cycle_method())
        self.root.bind('M', lambda e: self.cycle_method())
        self.root.bind('+', lambda e: self.increase_speed())
        self.root.bind('=', lambda e: self.increase_speed())
        self.root.bind('-', lambda e: self.decrease_speed())
        
        self.capture_thread = None
        self.running = True
    
    def cycle_method(self):
        # Cycle through aggregation methods
        methods = ['weighted_average', 'max_vote', 'consensus']
        current_idx = methods.index(self.aggregation_method)
        self.aggregation_method = methods[(current_idx + 1) % len(methods)]
        
        method_names = {
            'weighted_average': 'Weighted Avg',
            'max_vote': 'Max Vote',
            'consensus': 'Consensus'
        }
        self.stats_label.config(text=f"Method: {method_names[self.aggregation_method]}")
    
    def reset_accumulation(self):
        # Reset accumulated predictions (new location)
        self.accumulator.reset()
        self.stats_label.config(text="Reset! Views: 0")
        
        # Clear displays
        for widget in self.accumulated_frame.winfo_children():
            widget.destroy()
        for widget in self.current_frame.winfo_children():
            widget.destroy()
        
        tk.Label(self.accumulated_frame, 
                text="Move around to accumulate predictions...",
                font=('Arial', 10), bg='#0a0e27', fg='#95a5a6').pack(pady=20)
    
    def increase_speed(self):
        self.update_interval = max(0.5, self.update_interval - 0.5)
    
    def decrease_speed(self):
        self.update_interval = min(5.0, self.update_interval + 0.5)
    
    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.status_dot.config(text="⏸", fg='#ff6b6b')
        else:
            self.status_dot.config(text="▶", fg='#00ff88')
            if self.capture_thread is None or not self.capture_thread.is_alive():
                self.start_capture()
    
    def start_capture(self):
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.capture_thread.start()
    
    def capture_loop(self):
        while self.running:
            if not self.paused:
                try:
                    # Capture
                    if self.region:
                        screenshot = ImageGrab.grab(bbox=self.region)
                    else:
                        screenshot = ImageGrab.grab()
                    
                    # Get prediction probabilities
                    prediction_probs = self.predictor.predict(screenshot)
                    
                    # Add to accumulator
                    self.accumulator.add(prediction_probs)
                    
                    # Get both current and accumulated
                    current = self.accumulator.get_current(top_k=3)
                    accumulated = self.accumulator.get_accumulated(
                        top_k=5, 
                        method=self.aggregation_method
                    )
                    
                    # Update UI
                    self.root.after(0, self.update_results, current, accumulated)
                    
                except Exception as e:
                    print(f"Error: {e}")
                
                time.sleep(self.update_interval)
            else:
                time.sleep(0.1)
    
    def update_results(self, current, accumulated):
        # Update stats
        num_views = self.accumulator.get_sample_count()
        trend = self.accumulator.get_confidence_trend()
        trend_emoji = {"increasing": "📈", "decreasing": "📉", "stable": "➡️"}
        
        if accumulated:
            top_conf = accumulated[0][1]
            self.stats_label.config(
                text=f"Views: {num_views} | Top: {top_conf:.1f}% {trend_emoji[trend]}"
            )
        
        # Update ACCUMULATED (main display)
        for widget in self.accumulated_frame.winfo_children():
            widget.destroy()
        
        if accumulated:
            for i, (country, confidence) in enumerate(accumulated):
                if i == 0:
                    # Top accumulated guess - BIG and highlighted
                    bg = '#00d4ff'
                    fg = '#0a0e27'
                    size = 13
                    weight = 'bold'
                    height = 2
                else:
                    bg = '#1a2332'
                    fg = '#ecf0f1'
                    size = 10
                    weight = 'normal'
                    height = 1
                
                frame = tk.Frame(self.accumulated_frame, bg=bg, relief=tk.FLAT)
                frame.pack(fill=tk.X, pady=2 if i == 0 else 1)
                
                tk.Label(frame, text=f"{i+1}. {country}", 
                        font=('Arial', size, weight),
                        bg=bg, fg=fg, anchor='w',
                        height=height).pack(side=tk.LEFT, padx=8, fill=tk.X, expand=True)
                
                tk.Label(frame, text=f"{confidence:.1f}%",
                        font=('Arial', size, weight),
                        bg=bg, fg=fg, height=height).pack(side=tk.RIGHT, padx=8)
        
        # Update CURRENT (smaller display)
        for widget in self.current_frame.winfo_children():
            widget.destroy()
        
        if current:
            for i, (country, confidence) in enumerate(current):
                frame = tk.Frame(self.current_frame, bg='#1a1a2e', relief=tk.FLAT)
                frame.pack(fill=tk.X, pady=1)
                
                tk.Label(frame, text=f"{i+1}. {country}", 
                        font=('Arial', 8),
                        bg='#1a1a2e', fg='#95a5a6', anchor='w').pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
                
                tk.Label(frame, text=f"{confidence:.1f}%",
                        font=('Arial', 8),
                        bg='#1a1a2e', fg='#95a5a6').pack(side=tk.RIGHT, padx=5)
    
    def quit(self):
        self.running = False
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        print("\n" + "=" * 60)
        print("GeoGuessr AI")
        print("=" * 60)
        print("\nOverlay window opened!")
        print("\nHow it works:")
        print("  • AI analyzes each view as you move around")
        print("  • Combines all views for a smarter guess")
        print("  • More views = more confident prediction")
        print("\nControls:")
        print("  SPACE - Start/Stop predictions")
        print("  R     - Reset (new location)")
        print("  M     - Change aggregation method")
        print("  +/-   - Adjust update speed")
        print("  ESC   - Quit")
        print("\nPress SPACE to start, then move around in the game!")
        print("Press R when you move to a new location.")
        print("=" * 60 + "\n")
        
        self.root.mainloop()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='GeoGuessr AI - Smart accumulation mode')
    parser.add_argument('--model', type=str, default='best_model.pth')
    parser.add_argument('--labels', type=str, default='label_mapping.json')
    parser.add_argument('--interval', type=float, default=1.5)
    parser.add_argument('--no-select', action='store_true')
    parser.add_argument('--method', type=str, default='weighted_average',
                       choices=['weighted_average', 'max_vote', 'consensus'])
    
    args = parser.parse_args()
    
    # Check files
    if not Path(args.model).exists():
        print(f"Error: {args.model} not found!")
        sys.exit(1)
    
    if not Path(args.labels).exists():
        print(f"Error: {args.labels} not found!")
        sys.exit(1)
    
    # Region selection
    region = None
    if not args.no_select:
        print("\nSelect capture region...")
        selector = RegionSelector()
        region = selector.select()
    
    # Load model
    print("\nLoading model...")
    predictor = SmartPredictor(args.model, args.labels)
    
    # Run overlay
    overlay = SmartOverlay(predictor, region)
    overlay.update_interval = args.interval
    overlay.aggregation_method = args.method
    overlay.run()


if __name__ == "__main__":
    main()
