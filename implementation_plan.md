# Bangla OCR System — Implementation Plan

Build a complete Bangla OCR system using BanglaLekha-Isolated that trains a character recognition model, tracks experiments with MLflow, provides a Streamlit drawing UI, and runs inside Docker.

## User Review Required

> [!IMPORTANT]
> **Dataset Download**: The BanglaLekha-Isolated dataset (~188 MB ZIP) can be downloaded from:
> - **Kaggle** (recommended): `kaggle datasets download -d sifatmomen/banglalekha-isolated -p ./data --unzip` — requires `~/.kaggle/kaggle.json` credentials
> - **Mendeley Data** (official): [Manual download](https://data.mendeley.com/datasets/hf6sf8zrkc/2) — click "Download All"
>
> The `train.py` script will expect the dataset under `data/`. Should we include a helper download script that attempts Kaggle API download automatically?

> [!IMPORTANT]
> **Model Architecture**: The assignment allows CNN or transfer learning. I plan to implement **both** approaches to satisfy the "at least two MLflow runs" requirement:
> - **Run 1**: Custom CNN (lightweight, fast training)
> - **Run 2**: Transfer learning with MobileNetV2 (higher accuracy, pretrained on ImageNet)
>
> This naturally gives us two distinct experiment runs with different architectures to compare. Does this approach work for you?

> [!WARNING]
> **Word Segmentation Limitation**: BanglaLekha-Isolated contains *isolated* characters, not connected handwriting. The segmentation strategy will use **connected component analysis** on the canvas drawing — this works when characters are drawn with clear spacing between them, but will struggle with cursive/connected Bangla handwriting. The README will document this as a known limitation.

## Open Questions

1. **Python version**: Should we target Python 3.10+ (for Docker base image)?
2. **GPU training**: Should `train.py` auto-detect and use GPU if available, or should we keep it CPU-only for simplicity?
3. **Dataset subset**: Training on all 166K images across 84 classes may take time. Should we offer a `--subset` flag to train on fewer samples for quick testing?

---

## Dataset: BanglaLekha-Isolated

| Property | Value |
|---|---|
| Source | [Mendeley Data (v2)](https://data.mendeley.com/datasets/hf6sf8zrkc/2) / [Kaggle](https://www.kaggle.com/datasets/sifatmomen/banglalekha-isolated) |
| DOI | 10.17632/hf6sf8zrkc.2 |
| Total Images | 166,105 |
| Classes | 84 (11 vowels + 39 consonants + 10 numerals + 24 compound chars) |
| Organization | 84 folders, one per class |
| Image Format | PNG, grayscale, variable size (36×36 to 191×191) |
| ZIP Size | ~188 MB |
| File Naming | `{FormID}_{ClassNumber}.png` |

### Class-to-Character Mapping (labels.json)

The 84 classes map to Bangla Unicode characters in this order:

| Range | Category | Count | Characters |
|---|---|---|---|
| 0–10 | Vowels (স্বরবর্ণ) | 11 | অ, আ, ই, ঈ, উ, ঊ, ঋ, এ, ঐ, ও, ঔ |
| 11–49 | Consonants (ব্যঞ্জনবর্ণ) | 39 | ক, খ, গ, ঘ, ঙ, চ, ছ, জ, ঝ, ঞ, ট, ঠ, ড, ঢ, ণ, ত, থ, দ, ধ, ন, প, ফ, ব, ভ, ম, য, র, ল, শ, ষ, স, হ, ড়, ঢ়, য়, ৎ, ং, ঃ, ঁ |
| 50–59 | Numerals (সংখ্যা) | 10 | ০, ১, ২, ৩, ৪, ৫, ৬, ৭, ৮, ৯ |
| 60–83 | Compound Chars (যুক্তবর্ণ) | 24 | (defined in dataset metadata) |

> [!NOTE]
> The exact integer-to-character mapping depends on the folder names in the downloaded dataset. We will use `sorted(os.listdir('path/to/Images'))` to get the canonical order and save it as `labels.json` with format: `{"0": "অ", "1": "আ", ..., "83": "compound_char"}`.
> 
> **Approx ~2,000 samples per class** (relatively balanced after filtering).

---

## Proposed Changes

### 1. Dataset Preparation & Preprocessing

#### [NEW] [data/README.md](file:///Users/musakhan/Documents/Practice/BanglaOCR/data/README.md)
- Instructions on how to download and place the BanglaLekha-Isolated dataset
- Expected folder structure: `data/BanglaLekha-Isolated/0/`, `data/BanglaLekha-Isolated/1/`, ..., `data/BanglaLekha-Isolated/83/`

**Preprocessing Pipeline** (implemented in `train.py`):
1. **Load images** using `tf.keras.utils.image_dataset_from_directory`
2. **Resize** all images to **64×64** pixels (uniform input size)
3. **Convert to grayscale** (single channel) — images are already grayscale but may have varying channels
4. **Normalize** pixel values to [0, 1] range using `Rescaling(1./255)`
5. **Split**: 80% train / 10% validation / 10% test (stratified)
6. **Data augmentation** (training set only):
   - Random rotation (±10°)
   - Random translation (±10%)
   - Random zoom (±10%)
7. **Prefetch & cache** for performance using `tf.data.AUTOTUNE`

---

### 2. Training Script

#### [NEW] [train.py](file:///Users/musakhan/Documents/Practice/BanglaOCR/train.py)

**Architecture — Run 1: Custom CNN**
```
Input (64×64×1)
→ Conv2D(32, 3×3, ReLU) → BatchNorm → MaxPool(2×2)
→ Conv2D(64, 3×3, ReLU) → BatchNorm → MaxPool(2×2)
→ Conv2D(128, 3×3, ReLU) → BatchNorm → MaxPool(2×2)
→ Conv2D(256, 3×3, ReLU) → BatchNorm → MaxPool(2×2)
→ GlobalAveragePooling2D
→ Dense(256, ReLU) → Dropout(0.5)
→ Dense(84, Softmax)
```

**Architecture — Run 2: Transfer Learning (MobileNetV2)**
```
Input (64×64×3)  # MobileNetV2 requires 3 channels, so grayscale→RGB
→ MobileNetV2(weights='imagenet', include_top=False)  # Frozen base
→ GlobalAveragePooling2D
→ Dense(256, ReLU) → Dropout(0.5)
→ Dense(84, Softmax)
# Fine-tune: unfreeze last 20 layers after initial training
```

**Training Configuration**:
| Parameter | Run 1 (CNN) | Run 2 (MobileNetV2) |
|---|---|---|
| Optimizer | Adam | Adam |
| Learning Rate | 0.001 | 0.0001 |
| Batch Size | 64 | 32 |
| Epochs | 30 | 20 + 10 (fine-tune) |
| Loss | categorical_crossentropy | categorical_crossentropy |
| Callbacks | EarlyStopping, ModelCheckpoint, ReduceLROnPlateau | Same |

**MLflow Tracking** (both runs):
- **Parameters**: model_type, learning_rate, batch_size, epochs, optimizer, image_size, num_classes
- **Metrics** (per epoch): train_loss, train_accuracy, val_loss, val_accuracy
- **Artifacts**: model file (`model.keras`), `labels.json`, training history plot, confusion matrix plot, classification report
- **Tags**: run description, dataset version

**Output**:
- `models/model.keras` — best model (selected from better-performing run)
- `labels.json` — class index to Bangla character mapping
- MLflow runs logged to `artifacts/mlflow/`

**CLI Flags**:
```bash
python train.py                          # Train both runs
python train.py --model cnn              # Only custom CNN
python train.py --model mobilenet        # Only MobileNetV2
python train.py --epochs 10              # Override epochs
python train.py --data-dir ./data/BanglaLekha-Isolated
```

---

### 3. Prediction UI (Streamlit App)

#### [NEW] [app.py](file:///Users/musakhan/Documents/Practice/BanglaOCR/app.py)

**UI Layout**:
```
┌─────────────────────────────────────────────────┐
│  🔤 Bangla OCR — Handwritten Character Reader   │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌───────────────────────┐  ┌────────────────┐  │
│  │                       │  │  Predicted Word │  │
│  │    Drawing Canvas     │  │                │  │
│  │    (400×200 px)       │  │   বাংলা        │  │
│  │                       │  │                │  │
│  │                       │  │  Confidence:   │  │
│  │                       │  │  ব - 96.2%     │  │
│  └───────────────────────┘  │  া - 93.1%     │  │
│                             │  ং - 91.7%     │  │
│  [Predict] [Clear Canvas]   │  ল - 89.4%     │  │
│                             │  া - 93.1%     │  │
│                             └────────────────┘  │
│                                                 │
│  ┌────────────────────────────────────────────┐  │
│  │ Segmented Characters:                     │  │
│  │ [img1] [img2] [img3] [img4] [img5]       │  │
│  │  Top-3 predictions for each character     │  │
│  └────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

**Key Components**:
1. **Drawing Canvas**: Using `streamlit-drawable-canvas` library
   - White background, black stroke
   - Configurable stroke width
   - Landscape orientation (400×200) for word-level drawing
2. **Preprocessing Pipeline** (canvas → model input):
   - Convert canvas RGBA to grayscale
   - Apply binary threshold (Otsu's method)
   - Find contours / connected components
   - Sort components left-to-right
   - Extract bounding boxes for each character
   - Resize each character region to 64×64
   - Normalize to [0, 1]
3. **Word Segmentation Strategy**:
   - Use OpenCV `findContours` on the binarized canvas image
   - Filter contours by minimum area (remove noise)
   - Compute bounding boxes, sort left-to-right by x-coordinate
   - Merge overlapping/close bounding boxes (handles multi-stroke characters)
   - Extract each character as a separate image
4. **Prediction Display**:
   - Show segmented character images in a row
   - Per-character: top prediction + confidence score
   - Top-3 alternative predictions for each character
   - Combined predicted word displayed prominently

---

### 4. MLflow Integration

#### Tracking Configuration
- **Tracking URI**: `./artifacts/mlflow` (local file-based, no server needed)
- **Experiment Name**: `bangla-ocr-training`
- Minimum 2 runs (CNN vs MobileNetV2)
- Each run logs:
  - All hyperparameters
  - Training/validation metrics per epoch
  - Best model artifact
  - Label mapping file
  - Confusion matrix image
  - Classification report (text)

#### MLflow UI
```bash
mlflow ui --backend-store-uri ./artifacts/mlflow --host 0.0.0.0 --port 5000
```

---

### 5. Docker

#### [NEW] [Dockerfile](file:///Users/musakhan/Documents/Practice/BanglaOCR/Dockerfile)

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]
```

**Build & Run**:
```bash
docker build -t bangla-ocr-app:0.1 .
docker run -p 8501:8501 bangla-ocr-app:0.1
```

> [!NOTE]
> The Docker image only packages the Streamlit app + trained model. Training should be done outside Docker. The `models/model.keras` and `labels.json` must exist before building the image.

---

### 6. Requirements

#### [NEW] [requirements.txt](file:///Users/musakhan/Documents/Practice/BanglaOCR/requirements.txt)

```
tensorflow>=2.15.0
mlflow>=2.10.0
streamlit>=1.30.0
streamlit-drawable-canvas>=0.9.3
opencv-python-headless>=4.9.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.13.0
scikit-learn>=1.3.0
Pillow>=10.0.0
```

---

### 7. README

#### [NEW] [README.md](file:///Users/musakhan/Documents/Practice/BanglaOCR/README.md)

Comprehensive documentation covering:

1. **Project Overview** — What this project does
2. **Dataset Preparation** — Download instructions, expected structure
3. **Preprocessing** — Resize, normalize, augment pipeline
4. **Model Architecture** — CNN and MobileNetV2 details with diagrams
5. **Training Process** — How to run `train.py`, CLI options
6. **MLflow Tracking** — How experiments are tracked, how to view UI
7. **Word Segmentation Strategy** — Connected component analysis, bounding box extraction, limitations
8. **Streamlit UI Usage** — How to use the drawing canvas, what results mean
9. **Docker Build/Run** — Step-by-step commands
10. **Limitations** — Segmentation accuracy, compound characters, connected strokes
11. **Possible Improvements** — Sequence models (CRNN), attention mechanisms, better segmentation

---

### 8. Supporting Files

#### [NEW] [labels.json](file:///Users/musakhan/Documents/Practice/BanglaOCR/labels.json)
- Generated during training from the dataset folder names
- Maps class indices to Bangla Unicode characters
- Format: `{"0": "০", "1": "১", ..., "83": "compound_char"}`

#### [NEW] [.gitignore](file:///Users/musakhan/Documents/Practice/BanglaOCR/.gitignore)
```
data/
artifacts/mlflow/
models/
__pycache__/
*.pyc
.venv/
```

#### [NEW] [github_link.txt](file:///Users/musakhan/Documents/Practice/BanglaOCR/github_link.txt)
- Placeholder for the GitHub repository link

---

## Final Project Structure

```
BanglaOCR/
├── train.py                    # Training script with MLflow
├── app.py                      # Streamlit prediction UI
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker config for Streamlit app
├── README.md                   # Complete documentation
├── labels.json                 # Class index → Bangla character mapping
├── .gitignore                  # Exclude data, models, mlflow artifacts
├── github_link.txt             # GitHub repo link placeholder
├── models/
│   └── model.keras             # Best trained model
├── artifacts/
│   └── mlflow/                 # MLflow experiment tracking data
├── data/
│   ├── README.md               # Dataset download instructions
│   └── BanglaLekha-Isolated/   # Dataset (not committed)
│       ├── 0/                  # Class 0 images
│       ├── 1/                  # Class 1 images
│       └── ...83/              # Class 83 images
└── screenshots/
    ├── streamlit_app.png       # Screenshot of running app
    └── mlflow_experiment.png   # Screenshot of MLflow UI
```

---

## Verification Plan

### Automated Tests
1. **Training**: Run `python train.py --model cnn --epochs 2` with a small data subset to verify end-to-end training pipeline
2. **MLflow**: Verify runs are logged by checking `mlflow experiments search` output
3. **Model Loading**: Verify `model.keras` loads correctly and produces predictions of shape `(batch, 84)`
4. **Segmentation**: Test character segmentation on a synthetic test image with known character positions
5. **Docker**: Run `docker build -t bangla-ocr-app:0.1 . && docker run -d -p 8501:8501 bangla-ocr-app:0.1` and verify the app is accessible at `http://localhost:8501`

### Manual Verification
1. Draw a Bangla word on the Streamlit canvas and verify character segmentation + recognition
2. Take screenshots of the Streamlit app and MLflow UI for the `screenshots/` directory
3. Verify the README covers all required documentation sections
4. Test Docker build/run on a clean environment

---

## Execution Order

1. Create project scaffolding (`.gitignore`, `requirements.txt`, `data/README.md`)
2. Implement `train.py` with preprocessing, CNN model, and MLflow tracking
3. Add MobileNetV2 transfer learning as second run in `train.py`
4. Generate `labels.json` from dataset
5. Implement `app.py` with canvas, segmentation, and prediction
6. Create `Dockerfile`
7. Write comprehensive `README.md`
8. Test full pipeline
9. Capture screenshots
