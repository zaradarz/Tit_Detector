# Tit Detector 🐦

A simple machine learning project that uses a Convolutional Neural Network (CNN) to detect tit birds in images. This project includes tools for dataset creation, model training, and inference.

![Tit Detector Demo](Readme_animated.gif)

## Features

- **Dataset Creation**: Automatically scrape and prepare training data using DuckDuckGo search
- **CNN Training**: Train a custom CNN model to distinguish tit birds from other images
- **CLI Inference**: Command-line tool for quick image classification with confidence scores
- **Advanced Visualization**: Feature map visualization to understand what the model learns

## Project Structure

```
TitDetector/
├── scrape_and_prepare_dataset.py  # Dataset creation and preparation
├── train_model.py                 # CNN model training
├── tit_detector_cli.py            # Command-line inference tool
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd TitDetector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Create Dataset
Generate training data by scraping images from the web:
```bash
python scrape_and_prepare_dataset.py
```

This will create a `dataset/` folder with:
- `tits/` - Images of tit birds (parus major, great tit, blue tit, etc.)
- `not_tits/` - Images of landscapes, people, trees, buildings, etc.

### 2. Train the Model
Train the CNN model on your dataset:
```bash
python train_model.py
```

The script will:
- Train for 15 epochs with validation
- Save the best model as `tit_detector_cnn.pth`
- Generate a loss curve plot (`loss_curve.png`)

### 3. Run Inference
Test the trained model on new images:

**Basic usage:**
```bash
python tit_detector_cli.py --image /path/to/your/image.jpg
```

**With advanced feature visualization:**
```bash
python tit_detector_cli.py --image /path/to/your/image.jpg --advanced
```

**Examples:**
```bash
# Local file
python tit_detector_cli.py --image bird_photo.jpg

# URL
python tit_detector_cli.py --image https://example.com/bird.jpg

# Advanced mode with custom color range
python tit_detector_cli.py --image bird.jpg --advanced --range -3 3
```

## Model Architecture

The CNN uses a simple but effective architecture:
- **3 Convolutional blocks** with BatchNorm and ReLU
- **MaxPooling** for spatial dimension reduction
- **Dropout** for regularization
- **Fully connected layers** for final classification

Input: 128×128 RGB images  
Output: Binary classification (Tit / No Tit)

## Dependencies

- PyTorch - Deep learning framework
- torchvision - Computer vision utilities
- PIL/Pillow - Image processing
- requests - HTTP requests for image downloading
- tqdm - Progress bars
- matplotlib - Plotting and visualization
- duckduckgo-search - Web scraping for dataset creation


## License

This project is open source and available under the MIT License.

---

**Created by Zara Dar**  
**Assisted by Cursor AI**
