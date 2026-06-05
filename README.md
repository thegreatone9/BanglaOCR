# 🔤 Bangla OCR — Handwritten Character Recognition System

A complete Bangla OCR system that trains a character recognition model on the **BanglaLekha-Isolated** dataset, tracks experiments with **MLflow**, provides a **Streamlit** drawing UI for word-level recognition, and runs inside **Docker**.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Preparation](#dataset-preparation)
3. [Preprocessing Pipeline](#preprocessing-pipeline)
4. [Model Architectures](#model-architectures)
5. [Training](#training)
6. [MLflow Experiment Tracking](#mlflow-experiment-tracking)
7. [Word Segmentation Strategy](#word-segmentation-strategy)
8. [Streamlit UI](#streamlit-ui)
9. [Docker](#docker)
10. [Project Structure](#project-structure)
11. [Limitations & Known Issues](#limitations--known-issues)
12. [Possible Improvements](#possible-improvements)

---

## Project Overview

This project recognizes handwritten Bangla characters drawn on a canvas. The user can draw a **full Bangla word**, and the system will:

1. **Segment** the word into individual characters using a Matra-aware algorithm
2. **Classify** each character using a trained CNN/MobileNetV2 model
3. **Display** the combined predicted word with per-character confidence scores

The system supports **84 Bangla character classes**: 11 vowels, 39 consonants, 10 numerals, and 24 compound characters.

---

## Dataset Preparation

### BanglaLekha-Isolated

| Property       | Value                                                |
|----------------|------------------------------------------------------|
| Source          | [Mendeley Data](https://data.mendeley.com/datasets/hf6sf8zrkc/2) / [Kaggle](https://www.kaggle.com/datasets/sifatmomen/banglalekha-isolated) |
| Total Images   | ~166,105                                             |
| Classes        | 84                                                   |
| Image Format   | PNG, grayscale, variable size (36×36 to 191×191)     |
| Download Size  | ~188 MB                                              |

### Download

1. Visit: https://data.mendeley.com/datasets/hf6sf8zrkc/2
2. Click **"Download All"**
3. Extract into `data/` so the structure is:
```
data/
└── BanglaLekha-Isolated/
    ├── 0/    ← Class 0 images
    ├── 1/    ← Class 1 images
    └── ...83/
```

---

## Preprocessing Pipeline

All images pass through this pipeline before training:

| Step                | Details                                      |
|---------------------|----------------------------------------------|
| **Resize**          | All images resized to 64×64 pixels           |
| **Grayscale**       | Single channel (images are already grayscale)|
| **Normalize**       | Pixel values scaled to [0, 1]                |
| **Augmentation**    | Training set only: rotation (±5°), translation (±10%), zoom (±10%) |
| **Split**           | 80% training / 20% validation (seed=42)      |
| **Performance**     | Prefetch & cache using tf.data.AUTOTUNE      |

---

## Model Architectures

Two models are trained and compared via MLflow:

### Run 1: Custom CNN

```
Input (64×64×1)
→ Conv2D(32, 3×3) → BatchNorm → MaxPool(2×2)
→ Conv2D(64, 3×3) → BatchNorm → MaxPool(2×2)
→ Conv2D(128, 3×3) → BatchNorm → MaxPool(2×2)
→ Conv2D(256, 3×3) → BatchNorm → MaxPool(2×2)
→ GlobalAveragePooling2D
→ Dense(256, ReLU) → Dropout(0.5)
→ Dense(84, Softmax)
```

- **Optimizer**: Adam (lr=0.001)
- **Batch Size**: 64
- **Epochs**: 30 (with early stopping)

### Run 2: MobileNetV2 (Transfer Learning)

```
Input (64×64×3)  ← grayscale converted to RGB
→ MobileNetV2(pretrained on ImageNet, frozen)
→ GlobalAveragePooling2D
→ Dense(256, ReLU) → Dropout(0.5)
→ Dense(84, Softmax)
```

**Two-phase training:**
1. **Phase 1**: Frozen base, lr=0.0001, train top layers
2. **Phase 2**: Unfreeze last 20 layers, lr=0.00001, fine-tune

---

## Training

### Basic Usage

```bash
# Train both models
python train.py

# Train only the custom CNN
python train.py --model cnn

# Train only MobileNetV2
python train.py --model mobilenet

# Quick test with subset
python train.py --model cnn --epochs 5 --subset 500

# Custom data directory
python train.py --data-dir /path/to/dataset
```

### CLI Flags

| Flag            | Default                          | Description                           |
|-----------------|----------------------------------|---------------------------------------|
| `--model`       | `both`                           | `cnn`, `mobilenet`, or `both`         |
| `--epochs`      | `30`                             | Number of training epochs             |
| `--batch-size`  | `64`                             | Batch size                            |
| `--learning-rate`| `0.001`                         | Initial learning rate                 |
| `--subset`      | `None`                           | Use N samples per class (quick test)  |
| `--data-dir`    | `./data/BanglaLekha-Isolated`    | Path to dataset                       |
| `--output-dir`  | `./models`                       | Where to save the trained model       |

### Output

- `models/model.keras` — Best trained model
- `labels.json` — Class index → Bangla character mapping
- `artifacts/mlflow/` — MLflow experiment data

---

## MLflow Experiment Tracking

Each training run logs the following to MLflow:

| Category       | What's Logged                                                |
|----------------|--------------------------------------------------------------|
| **Parameters** | model_type, learning_rate, batch_size, epochs, image_size, num_classes |
| **Metrics**    | train_loss, train_accuracy, val_loss, val_accuracy (per epoch)|
| **Artifacts**  | model.keras, labels.json, confusion matrix, classification report |
| **Tags**       | dataset name, framework version                              |

### Viewing the MLflow UI

```bash
mlflow ui --backend-store-uri ./artifacts/mlflow --host 0.0.0.0 --port 5000
```

Then open http://localhost:5000 in your browser to compare runs.

---

## Word Segmentation Strategy

The system uses a **Matra-aware 3-phase** segmentation approach to handle connected Bangla characters. Because Bangla words are typically joined by a continuous top line (the *Matra*), simple character cropping fails on full words. Here is how our system breaks down a drawn word into individual characters for the AI to identify:

### Phase 1 — Matra Detection & Removal
The Matra connects multiple letters into a single visual "island" of ink. 
To separate the letters, the `detect_and_remove_matra()` function searches for the Matra by calculating the **horizontal projection profile** (summing the ink pixels row by row). The Matra appears as a massive spike in pixel density near the top. Once found, this horizontal line is temporarily erased from a working copy of the image, breaking the physical connection between the letters beneath it.

### Phase 2 — Vertical Projection Segmentation
With the top connecting line erased, the letters are now free-standing. 
The `segment_by_vertical_projection()` function calculates the **vertical projection profile** (summing ink column by column). Natural spaces between characters appear as completely empty columns (gaps). The algorithm slices the image vertically at these empty gaps, effectively separating the word into individual character blocks.

### Phase 3 — Connected Component Analysis (CCA) Fallback
Not all Bangla characters have a Matra (e.g., 'ও', 'এ'). If Phase 1 detects no Matra, vertical slicing might be unreliable.
In this case, the system falls back to `segment_by_contours()`. It uses Connected Component Analysis (finding isolated "islands" of ink) to draw tight bounding boxes around every shape. It filters out tiny specks of noise and uses `merge_close_boxes()` to reconnect characters that are made of multiple disconnected strokes (like dots or curls).

### Post-Processing & Identification
No matter which phase was used to cut the character, the resulting cropped image is sent to `preprocess_char()`. 
This function acts as the final cleaner: it centers the character, pads it into a perfect square, and resizes it to a clean 64×64 pixel image. 

Finally, these prepared 64×64 images are sent to the trained **CNN / MobileNetV2** model. The model calculates the probability of each image belonging to one of the 84 trained Bangla character classes, and the highest probability is returned as the final predicted letter!

---

## Streamlit UI

### Running the App

```bash
streamlit run app.py
```

### Features

- **Drawing Canvas**: 500×200 pixel landscape canvas for word-level drawing
- **Adjustable Stroke Width**: Configure pen thickness in the sidebar
- **Segmented Character Display**: See how each character was segmented
- **Per-Character Predictions**: Top prediction with confidence + top-3 alternatives
- **Combined Word Output**: Full predicted word displayed prominently

### Usage

1. Draw a Bangla word on the canvas
2. Click **"Predict"**
3. View the segmented characters and predicted word

---

## Docker

### Prerequisites

Before building the Docker image, ensure you have:
- A trained model at `models/model.keras`
- A `labels.json` file in the project root

### Build & Run

```bash
# Build the image
docker build -t bangla-ocr-app:0.1 .

# Run the container
docker run -p 8501:8501 bangla-ocr-app:0.1
```

Then open http://localhost:8501 in your browser.

> **Note**: The Docker image only packages the inference app (Streamlit + model). Training should be done outside Docker on your host machine (ideally with GPU).

---

## Project Structure

```
BanglaOCR/
├── train.py                    # Training script with MLflow tracking
├── app.py                      # Streamlit prediction UI
├── run.sh                      # Start/stop helper (Streamlit + MLflow)
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker config (inference only)
├── README.md                   # This file
├── labels.json                 # Class index → Bangla character mapping
├── .gitignore                  # Exclude data, models, artifacts
├── .streamlit/config.toml      # Streamlit theme/server config
├── github_link.txt             # GitHub repo link
├── artifacts/
│   └── mlflow.db               # MLflow experiment tracking database
├── data/
│   ├── README.md               # Dataset download instructions
│   └── BanglaLekha-Isolated/   # Dataset (not committed)
│       ├── 0/ ... 83/
└── screenshots/
    ├── streamlit_app.png       # UI screenshot
    └── mlflow_experiment.png   # MLflow screenshot
```

---

## Limitations & Known Issues

| Limitation | Root Cause | Impact |
|---|---|---|
| **Connected characters may mis-segment** | Matra removal is heuristic-based; complex ligatures may not separate cleanly | Some compound characters may be split incorrectly |
| **Compound characters** | Dataset has 24 compounds, but Bangla has 300+ | Many real-world compounds will be unrecognized |
| **No vowel sign handling** | Vowel signs (কার) are trained as separate classes but appear attached to consonants in real writing | Segmentation may not separate vowel signs correctly |
| **Canvas drawing ≠ real handwriting** | Mouse/touch drawing differs from pen-on-paper | Model accuracy may be lower on canvas-drawn characters |
| **Single model for inference** | Docker image uses the last trained model | Need to manually select best model from MLflow runs |

---

## Possible Improvements

1. **CRNN + CTC**: Replace segmentation-based approach with a sequence model that handles connected text directly
2. **Attention Mechanisms**: Add attention layers for better compound character recognition
3. **Data Augmentation**: Elastic deformation, erosion/dilation to simulate handwriting variation
4. **Ensemble Models**: Combine predictions from CNN and MobileNetV2
5. **Web Deployment**: Deploy to cloud (GCP, AWS) with a production-grade server
6. **Mobile Support**: Add touch-optimized canvas for mobile browsers
