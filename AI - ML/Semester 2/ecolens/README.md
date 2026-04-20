# EcoLens — Intelligent Waste Classification & Pollution Alert System

```
  ███████╗ ██████╗ ██████╗ ██╗     ███████╗███╗   ██╗███████╗
  ██╔════╝██╔════╝██╔═══██╗██║     ██╔════╝████╗  ██║██╔════╝
  █████╗  ██║     ██║   ██║██║     █████╗  ██╔██╗ ██║███████╗
  ██╔══╝  ██║     ██║   ██║██║     ██╔══╝  ██║╚██╗██║╚════██║
  ███████╗╚██████╗╚██████╔╝███████╗███████╗██║ ╚████║███████║
  ╚══════╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝
```

> AI/ML Project — Intelligent Waste Classification & Pollution Alert System

---

## 🌿 Overview

EcoLens is an AI-powered waste management system that:
- **Classifies** waste images into 6 categories using MobileNetV2
- **Labels** waste as dry/wet and recyclable/non-recyclable
- **Color-codes** each category for easy bin identification
- **Computes** a Pollution Index from batch images
- **Alerts** municipal/panchayat authorities via email when pollution is critical
- **Visualizes** analytics with interactive charts

---

## 📂 Project Structure

```
ecolens/
├── app.py                    ← Main Streamlit app (run this)
├── requirements.txt
├── .env.example              ← Copy to .env
├── download_dataset.py       ← Downloads TrashNet dataset
│
├── config/
│   └── waste_taxonomy.py     ← Waste categories, colors, thresholds
│
├── model/
│   ├── train.py              ← MobileNetV2 training pipeline
│   └── predict.py            ← Inference engine
│
├── utils/
│   └── alert_system.py       ← Email alert system
│
├── data/                     ← Created by download_dataset.py
│   ├── raw/
│   └── processed/
│       ├── train/
│       ├── val/
│       └── test/
│
├── models/                   ← Created by train.py
│   ├── ecolens_final.h5
│   ├── ecolens_best.h5
│   ├── metadata.json
│   └── training_history.png
│
└── logs/
    └── alerts.jsonl    

---

## 🚀 Quick Start

```  
NOTE: Python version `3.10.x` required for this project  

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download TrashNet Dataset
```bash
# Auto-download (~150MB from GitHub)
python download_dataset.py

# Or create sample data for testing:
python download_dataset.py --sample
```

### 3. Train the Model
```bash
python model/train.py
# Training takes ~30-60 min on CPU, ~5-10 min on GPU
# Expected accuracy: ~88-92%
```

### 4. Launch Web App
```bash
streamlit run app.py
```

---

## 🗂️ Dataset: TrashNet

| Class | Images | Color |
|-------|--------|-------|
| 📦 Cardboard | 403 | Blue |
| 🍾 Glass | 501 | Green |
| 🥫 Metal | 410 | Purple |
| 📄 Paper | 594 | Orange |
| 🧴 Plastic | 482 | Red |
| 🗑️ Trash | 137 | Grey |

**Source:** https://github.com/garythung/trashnet (Gary Thung & Mindy Yang, Stanford)

---

## 📧 Email Alerts

1. Enable 2-Step Verification on your Google account
2. Create an **App Password** (Google Account → Security → App Passwords)
3. Enter it in the EcoLens sidebar (never use your real password)
4. Set the authority's email and your details

Alerts are sent when the Pollution Index reaches **HIGH** (>55%) or **CRITICAL** (>75%).

---

## 🎯 Pollution Index

| Level | Score | Action |
|-------|-------|--------|
| 🟢 Low | 0-35% | No action needed |
| 🟡 Moderate | 35-55% | Monitor the area |
| 🟠 High | 55-75% | Alert authority |
| 🔴 Critical | 75-100% | Immediate action + Escalate |

---

## 🏗️ Model Architecture

- **Base:** MobileNetV2 (ImageNet pretrained)
- **Head:** GAP → BN → Dense(512) → Dropout(0.4) → Dense(256) → Dropout(0.3) → Dense(6, softmax)
- **Training:** 2-phase (frozen base → fine-tune top 30 layers)
- **Augmentation:** rotation, shift, flip, zoom, brightness

---

*Built for AI/ML Coursework | EcoLens v1.0*
