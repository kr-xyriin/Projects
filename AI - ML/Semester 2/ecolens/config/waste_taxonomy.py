# config/waste_taxonomy.py
# EcoLens — Waste Classification Taxonomy

WASTE_TAXONOMY = {
    "cardboard": {
        "display_name": "Cardboard",
        "moisture_type": "dry",
        "recyclable": True,
        "color_hex": "#4A90D9",        # Blue — dry recyclable
        "color_name": "Blue",
        "bin_color": "Blue Bin (Dry Recyclable)",
        "examples": ["boxes", "cartons", "packaging"],
        "disposal_tip": "Flatten boxes before disposal. Keep dry.",
        "icon": "📦",
        "pollution_weight": 0.3,       # Low pollution risk
    },
    "glass": {
        "display_name": "Glass",
        "moisture_type": "dry",
        "recyclable": True,
        "color_hex": "#27AE60",        # Green — glass recyclable
        "color_name": "Green",
        "bin_color": "Green Bin (Glass Recyclable)",
        "examples": ["bottles", "jars", "window glass"],
        "disposal_tip": "Rinse before recycling. Do not break.",
        "icon": "🍾",
        "pollution_weight": 0.4,
    },
    "metal": {
        "display_name": "Metal",
        "moisture_type": "dry",
        "recyclable": True,
        "color_hex": "#8E44AD",        # Purple — metal recyclable
        "color_name": "Purple",
        "bin_color": "Purple Bin (Metal Recyclable)",
        "examples": ["cans", "tins", "foil", "scrap metal"],
        "disposal_tip": "Crush cans to save space. Remove food residue.",
        "icon": "🥫",
        "pollution_weight": 0.5,
    },
    "paper": {
        "display_name": "Paper",
        "moisture_type": "dry",
        "recyclable": True,
        "color_hex": "#F39C12",        # Orange — paper recyclable
        "color_name": "Orange",
        "bin_color": "Orange Bin (Paper Recyclable)",
        "examples": ["newspapers", "magazines", "office paper", "books"],
        "disposal_tip": "Keep dry. Remove staples and plastic covers.",
        "icon": "📄",
        "pollution_weight": 0.2,
    },
    "plastic": {
        "display_name": "Plastic",
        "moisture_type": "dry",
        "recyclable": False,           # Conditionally recyclable — defaulting to non-recyclable for awareness
        "color_hex": "#E74C3C",        # Red — plastic non-recyclable warning
        "color_name": "Red",
        "bin_color": "Red Bin (Plastic — Check Recycling Code)",
        "examples": ["bottles", "bags", "containers", "packaging"],
        "disposal_tip": "Check recycling code (1-7). Avoid single-use plastics.",
        "icon": "🧴",
        "pollution_weight": 0.9,       # High pollution risk
    },
    "trash": {
        "display_name": "General Trash",
        "moisture_type": "wet",
        "recyclable": False,
        "color_hex": "#7F8C8D",        # Grey — general waste
        "color_name": "Grey",
        "bin_color": "Grey Bin (General / Wet Waste)",
        "examples": ["food scraps", "soiled items", "mixed waste", "organic waste"],
        "disposal_tip": "Segregate organic waste for composting when possible.",
        "icon": "🗑️",
        "pollution_weight": 0.8,
    },
}

# Pollution thresholds for alert system
POLLUTION_THRESHOLDS = {
    "low": {"min": 0.0, "max": 0.35, "label": "Low Pollution", "color": "#27AE60", "emoji": "🟢"},
    "moderate": {"min": 0.35, "max": 0.55, "label": "Moderate Pollution", "color": "#F39C12", "emoji": "🟡"},
    "high": {"min": 0.55, "max": 0.75, "label": "High Pollution", "color": "#E67E22", "emoji": "🟠"},
    "critical": {"min": 0.75, "max": 1.0, "label": "CRITICAL Pollution", "color": "#E74C3C", "emoji": "🔴"},
}

# Authority contact map by area type
AUTHORITY_CONTACTS = {
    "urban": {
        "title": "Municipal Corporation",
        "role": "Commissioner / Sanitation Officer",
        "urgency": "high",
    },
    "semi_urban": {
        "title": "Municipal Council / Nagar Panchayat",
        "role": "Executive Officer",
        "urgency": "medium",
    },
    "rural": {
        "title": "Gram Panchayat",
        "role": "Sarpanch / Ward Member",
        "urgency": "medium",
    },
    "industrial": {
        "title": "Pollution Control Board",
        "role": "Regional Officer",
        "urgency": "critical",
    },
}

MODEL_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
