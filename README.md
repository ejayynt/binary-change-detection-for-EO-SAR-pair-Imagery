# Satellite AI Research Intern Technical Assignment

## Project Description

This repository contains the implementation for the GalaxEye Space Satellite AI Research Intern technical assignment.

The project focuses on building a deep learning pipeline for satellite image analysis using PyTorch. The implementation includes:

- Exploratory Data Analysis (EDA)
- Data preprocessing and augmentation
- Model architecture definition
- Training and validation pipeline
- Evaluation and inference scripts
- Reproducible experiment configuration

The codebase is structured to be modular, reproducible, and easy to extend for further experimentation.

---

# Repository Structure

```bash
.
├── EDA.py                 # Exploratory data analysis and visualization
├── train.py               # Training pipeline and experiment runner
├── eval.py                # Evaluation and inference script
├── model.py               # Model architecture definition
├── proposal.md            # Project proposal and methodology
├── requirements.txt       # Python dependencies
├── config.yaml            # Experiment configuration and hyperparameters
├── checkpoints/           # Saved model checkpoints
├── outputs/               # Logs, predictions, and evaluation outputs
└── README.md              # Project documentation
```

---

# Requirements

## Python Version

- Python 3.10+

## Dependencies

All dependencies are listed in `requirements.txt` with pinned versions.

Install dependencies using:

```bash
pip install -r requirements.txt
```

Example dependencies:

```txt
torch==2.2.2
torchvision==0.17.2
numpy==1.26.4
opencv-python==4.9.0.80
scikit-learn==1.4.2
pandas==2.2.2
matplotlib==3.8.4
tqdm==4.66.2
pyyaml==6.0.1
albumentations==1.4.4
```

---

# Environment Setup

## Option 1 — Using Conda

### Step 1: Create Environment

```bash
conda create -n satellite-ai python=3.10 -y
```

### Step 2: Activate Environment

```bash
conda activate satellite-ai
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Option 2 — Using venv

### Step 1: Create Virtual Environment

```bash
python -m venv venv
```

### Step 2: Activate Environment

#### Linux / MacOS

```bash
source venv/bin/activate
```

#### Windows

```bash
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Configuration

All experiment hyperparameters are stored in:

```bash
config.yaml
```

Example parameters logged:

```yaml
learning_rate: 0.0001
batch_size: 16
epochs: 50
optimizer: adamw
loss_function: cross_entropy
image_size: 224
random_seed: 42

augmentations:
  horizontal_flip: true
  vertical_flip: true
  rotation: 15
```

The configuration file ensures complete reproducibility of experiments.

---

# Dataset Structure

After downloading and extracting the dataset, organize it in the following structure:

```bash
dataset/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
├── val/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── test/
    ├── class_1/
    ├── class_2/
    └── ...
```

If the task is segmentation-based:

```bash
dataset/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

Update dataset paths in `config.yaml` if required.

---

# Training

To train the model from scratch:

```bash
python train.py --config config.yaml
```

Training outputs:

- Model checkpoints → `checkpoints/`
- Logs and metrics → `outputs/`

---

# Evaluation

To evaluate a trained model on the test dataset:

```bash
python eval.py \
    --data_path /path/to/test \
    --weights /path/to/checkpoint.pth
```

Example:

```bash
python eval.py \
    --data_path dataset/test \
    --weights checkpoints/best_model.pth
```

Evaluation metrics will be printed to the console and optionally saved in `outputs/`.

---

# Exploratory Data Analysis

To run exploratory data analysis:

```bash
python EDA.py
```

This script generates:

- Dataset distribution analysis
- Sample visualizations
- Class balance inspection
- Statistical summaries

---

# Model Weights

Download trained model checkpoints from:

```text
<INSERT_PUBLIC_CHECKPOINT_LINK_HERE>
```

Example hosting platforms:

- Google Drive
- HuggingFace Hub
- Dropbox

---

# Results

## Validation Metrics

| Metric | Score |
|--------|--------|
| Accuracy | XX.XX |
| Precision | XX.XX |
| Recall | XX.XX |
| F1 Score | XX.XX |

## Test Metrics

| Metric | Score |
|--------|--------|
| Accuracy | XX.XX |
| Precision | XX.XX |
| Recall | XX.XX |
| F1 Score | XX.XX |

> Replace placeholder values with your final reported metrics.

---

# Reproducibility

To ensure reproducibility:

- Fixed random seed is used
- All hyperparameters are stored in `config.yaml`
- Dependency versions are pinned
- Deterministic training settings are enabled where applicable

---

# Citation / References

## Papers

1. Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*
2. He et al., *Deep Residual Learning for Image Recognition*
3. Howard et al., *Searching for MobileNetV3*

## Libraries / Codebases

- PyTorch
- torchvision
- Albumentations
- timm

---

# GitHub Repository

```text
<INSERT_GITHUB_REPOSITORY_LINK_HERE>
```
