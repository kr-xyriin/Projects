"""
model/predict.py
EcoLens — Waste Classification Inference Engine
Handles single image and batch predictions with full metadata.
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
import io

# Lazy TF import to speed up cold starts
_model = None
_model_path = None

MODEL_DIR = Path("models")
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMAGE_SIZE = (224, 224)


def load_model(model_path: str = None):
    """Load model lazily."""
    global _model, _model_path
    import tensorflow as tf
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    if model_path is None:
        # Try to find best available model
        candidates = [
            MODEL_DIR / "ecolens_final.h5",
            MODEL_DIR / "ecolens_best.h5",
            MODEL_DIR / "ecolens_phase1.h5",
        ]
        for candidate in candidates:
            if candidate.exists():
                model_path = str(candidate)
                break

    if model_path is None or not Path(model_path).exists():
        raise FileNotFoundError(
            "No trained model found. Please run: python model/train.py\n"
            "Or download a pretrained model."
        )

    if _model is None or _model_path != model_path:
        _model = tf.keras.models.load_model(model_path)
        _model_path = model_path
        print(f"✅ Model loaded from: {model_path}")

    return _model


def preprocess_image(image_input) -> np.ndarray:
    """
    Preprocess image for inference.
    Accepts: PIL.Image, bytes, file path, or numpy array.
    """
    if isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, (str, Path)):
        img = Image.open(image_input)
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input.astype('uint8'))
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise ValueError(f"Unsupported image type: {type(image_input)}")

    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize and normalize
    img = img.resize(IMAGE_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # Add batch dim


def predict_single(image_input, model=None) -> dict:
    """
    Predict waste class for a single image.
    Returns full classification metadata from waste_taxonomy.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config.waste_taxonomy import WASTE_TAXONOMY

    if model is None:
        model = load_model()

    arr = preprocess_image(image_input)
    probs = model.predict(arr, verbose=0)[0]

    predicted_idx = int(np.argmax(probs))
    predicted_class = CLASSES[predicted_idx]
    confidence = float(probs[predicted_idx])

    # Top-3 predictions
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [
        {"class": CLASSES[i], "confidence": float(probs[i]),
         "display_name": WASTE_TAXONOMY[CLASSES[i]]["display_name"]}
        for i in top3_idx
    ]

    # Get taxonomy metadata
    meta = WASTE_TAXONOMY[predicted_class]

    return {
        "predicted_class": predicted_class,
        "display_name": meta["display_name"],
        "confidence": confidence,
        "confidence_pct": f"{confidence * 100:.1f}%",
        "moisture_type": meta["moisture_type"],
        "recyclable": meta["recyclable"],
        "color_hex": meta["color_hex"],
        "color_name": meta["color_name"],
        "bin_color": meta["bin_color"],
        "disposal_tip": meta["disposal_tip"],
        "icon": meta["icon"],
        "pollution_weight": meta["pollution_weight"],
        "top3_predictions": top3,
        "all_probs": {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
    }


def predict_batch(image_list, model=None, batch_size=16) -> list:
    """
    Predict for a list of images. Returns list of prediction dicts + pollution score.
    """
    if model is None:
        model = load_model()

    results = []
    for img in image_list:
        try:
            result = predict_single(img, model=model)
            results.append(result)
        except Exception as e:
            results.append({"error": str(e), "predicted_class": "unknown"})

    return results


def compute_pollution_score(predictions: list) -> dict:
    """
    Compute area pollution index from batch predictions.
    Returns score, level, breakdown, and alert recommendation.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config.waste_taxonomy import POLLUTION_THRESHOLDS, WASTE_TAXONOMY

    if not predictions:
        return {"score": 0.0, "level": "low", "breakdown": {}}

    valid = [p for p in predictions if "error" not in p]
    if not valid:
        return {"score": 0.0, "level": "low", "breakdown": {}}

    # Weighted pollution score
    total_weight = sum(p["pollution_weight"] for p in valid)
    avg_score = total_weight / len(valid)

    # Class breakdown
    from collections import Counter
    class_counts = Counter(p["predicted_class"] for p in valid)
    breakdown = dict(class_counts)

    # Determine level
    level_key = "low"
    for key, threshold in POLLUTION_THRESHOLDS.items():
        if threshold["min"] <= avg_score <= threshold["max"]:
            level_key = key
            break

    threshold_info = POLLUTION_THRESHOLDS[level_key]
    should_alert = level_key in ("high", "critical")

    # Compute percentages
    total = len(valid)
    recyclable_count = sum(1 for p in valid if p.get("recyclable", False))
    dry_count = sum(1 for p in valid if p.get("moisture_type") == "dry")

    return {
        "score": round(avg_score, 3),
        "score_pct": f"{avg_score * 100:.1f}%",
        "level": level_key,
        "label": threshold_info["label"],
        "color": threshold_info["color"],
        "emoji": threshold_info["emoji"],
        "should_alert": should_alert,
        "total_items": total,
        "recyclable_count": recyclable_count,
        "recyclable_pct": round(recyclable_count / total * 100, 1) if total else 0,
        "dry_count": dry_count,
        "wet_count": total - dry_count,
        "breakdown": breakdown,
    }


def get_model_info() -> dict:
    """Get model metadata if available."""
    meta_path = MODEL_DIR / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {
        "model_name": "EcoLens Waste Classifier",
        "architecture": "MobileNetV2 Transfer Learning",
        "classes": CLASSES,
        "note": "Model not yet trained. Run: python model/train.py"
    }
