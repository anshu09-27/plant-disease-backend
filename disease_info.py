"""
disease_info.py

Maps the 38 PlantVillage class labels (in the standard alphabetical order
used by MobileNetV2 training pipelines) to:
  - a human-readable display name
  - whether the plant is healthy
  - a short treatment / remedy recommendation

IMPORTANT: The order of CLASS_NAMES must exactly match the order of the
output classes your model was trained on (i.e. the order Keras'
ImageDataGenerator / image_dataset_from_directory assigned during training,
usually alphabetical by folder name). If your training notebook printed a
class_indices dict, double check it matches this list before deploying.
"""

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# key = raw class name above -> {display, healthy, treatment}
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "display_name": "Apple Scab",
        "healthy": False,
        "treatment": "Remove and destroy fallen leaves to reduce spore source. Apply fungicides (captan or myclobutanil) starting at bud break, repeating every 7-10 days in wet weather. Prune for better air circulation.",
    },
    "Apple___Black_rot": {
        "display_name": "Apple Black Rot",
        "healthy": False,
        "treatment": "Prune out dead/cankered wood and mummified fruit. Apply fungicides (captan or thiophanate-methyl) during the growing season. Remove nearby wild/abandoned apple trees that can harbor infection.",
    },
    "Apple___Cedar_apple_rust": {
        "display_name": "Cedar Apple Rust",
        "healthy": False,
        "treatment": "Remove nearby juniper/cedar hosts if feasible. Apply protectant fungicides (myclobutanil or mancozeb) from pink bud stage through petal fall. Choose resistant apple varieties for new plantings.",
    },
    "Apple___healthy": {
        "display_name": "Healthy Apple",
        "healthy": True,
        "treatment": "No treatment needed. Maintain regular watering, balanced fertilization, and periodic pruning to keep the tree healthy.",
    },
    "Blueberry___healthy": {
        "display_name": "Healthy Blueberry",
        "healthy": True,
        "treatment": "No treatment needed. Maintain acidic, well-drained soil (pH 4.5-5.5) and consistent moisture.",
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "display_name": "Cherry Powdery Mildew",
        "healthy": False,
        "treatment": "Apply sulfur or potassium bicarbonate based fungicides at first sign of disease. Improve air circulation through pruning and avoid excess nitrogen fertilization.",
    },
    "Cherry_(including_sour)___healthy": {
        "display_name": "Healthy Cherry",
        "healthy": True,
        "treatment": "No treatment needed. Continue routine care with adequate sunlight and pruning.",
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "display_name": "Corn Gray Leaf Spot",
        "healthy": False,
        "treatment": "Rotate crops away from corn for at least one season. Use resistant hybrids. Apply foliar fungicides (strobilurin or triazole based) if disease pressure is high.",
    },
    "Corn_(maize)___Common_rust_": {
        "display_name": "Corn Common Rust",
        "healthy": False,
        "treatment": "Plant resistant hybrids. Apply fungicides (azoxystrobin or propiconazole) if rust appears early and conditions favor spread. Rust is usually minor in resistant varieties.",
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "display_name": "Corn Northern Leaf Blight",
        "healthy": False,
        "treatment": "Use resistant hybrids and rotate crops. Apply fungicide (e.g. mancozeb or strobilurins) at early disease onset. Till under crop debris after harvest to reduce inoculum.",
    },
    "Corn_(maize)___healthy": {
        "display_name": "Healthy Corn",
        "healthy": True,
        "treatment": "No treatment needed. Maintain proper spacing, fertility, and irrigation.",
    },
    "Grape___Black_rot": {
        "display_name": "Grape Black Rot",
        "healthy": False,
        "treatment": "Remove mummified berries and infected canes during dormant pruning. Apply fungicides (mancozeb or myclobutanil) starting at bud break through veraison.",
    },
    "Grape___Esca_(Black_Measles)": {
        "display_name": "Grape Esca (Black Measles)",
        "healthy": False,
        "treatment": "No fully effective chemical cure exists. Prune out and destroy infected wood, avoid large pruning wounds during wet weather, and apply wound-protectant products after pruning.",
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "display_name": "Grape Leaf Blight (Isariopsis Leaf Spot)",
        "healthy": False,
        "treatment": "Remove and destroy infected leaves. Apply copper-based fungicides and ensure good canopy airflow through proper pruning and trellising.",
    },
    "Grape___healthy": {
        "display_name": "Healthy Grape",
        "healthy": True,
        "treatment": "No treatment needed. Maintain a regular pruning and canopy management schedule.",
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "display_name": "Citrus Greening (Huanglongbing)",
        "healthy": False,
        "treatment": "No cure exists. Remove and destroy infected trees to prevent spread. Control the Asian citrus psyllid vector with approved insecticides and use certified disease-free nursery stock.",
    },
    "Peach___Bacterial_spot": {
        "display_name": "Peach Bacterial Spot",
        "healthy": False,
        "treatment": "Plant resistant varieties. Apply copper-based bactericides during dormancy and early season. Avoid overhead irrigation and excessive nitrogen.",
    },
    "Peach___healthy": {
        "display_name": "Healthy Peach",
        "healthy": True,
        "treatment": "No treatment needed. Continue regular pruning, fertilization, and pest monitoring.",
    },
    "Pepper,_bell___Bacterial_spot": {
        "display_name": "Bell Pepper Bacterial Spot",
        "healthy": False,
        "treatment": "Use disease-free seed/transplants. Apply copper-based bactericides, avoid overhead watering, and rotate crops with non-solanaceous plants for 2-3 years.",
    },
    "Pepper,_bell___healthy": {
        "display_name": "Healthy Bell Pepper",
        "healthy": True,
        "treatment": "No treatment needed. Maintain consistent watering and balanced fertilization.",
    },
    "Potato___Early_blight": {
        "display_name": "Potato Early Blight",
        "healthy": False,
        "treatment": "Apply fungicides (chlorothalonil or mancozeb) at first symptoms. Rotate crops, avoid overhead irrigation, and remove infected plant debris after harvest.",
    },
    "Potato___Late_blight": {
        "display_name": "Potato Late Blight",
        "healthy": False,
        "treatment": "Apply fungicides (chlorothalonil or copper-based) promptly, as this disease spreads fast. Destroy infected plants, avoid overhead watering, and use certified disease-free seed potatoes.",
    },
    "Potato___healthy": {
        "display_name": "Healthy Potato",
        "healthy": True,
        "treatment": "No treatment needed. Maintain proper hilling, watering, and crop rotation practices.",
    },
    "Raspberry___healthy": {
        "display_name": "Healthy Raspberry",
        "healthy": True,
        "treatment": "No treatment needed. Prune canes annually and ensure good air circulation.",
    },
    "Soybean___healthy": {
        "display_name": "Healthy Soybean",
        "healthy": True,
        "treatment": "No treatment needed. Maintain crop rotation and monitor for early pest/disease signs.",
    },
    "Squash___Powdery_mildew": {
        "display_name": "Squash Powdery Mildew",
        "healthy": False,
        "treatment": "Apply sulfur, potassium bicarbonate, or neem oil at first sign of white powdery patches. Improve air circulation and avoid overhead watering late in the day.",
    },
    "Strawberry___Leaf_scorch": {
        "display_name": "Strawberry Leaf Scorch",
        "healthy": False,
        "treatment": "Remove infected leaves after harvest. Apply fungicides (captan or myclobutanil) during the growing season. Avoid overhead irrigation and ensure good plant spacing.",
    },
    "Strawberry___healthy": {
        "display_name": "Healthy Strawberry",
        "healthy": True,
        "treatment": "No treatment needed. Maintain mulching, consistent watering, and periodic runner management.",
    },
    "Tomato___Bacterial_spot": {
        "display_name": "Tomato Bacterial Spot",
        "healthy": False,
        "treatment": "Use certified disease-free seed/transplants. Apply copper-based bactericides, avoid overhead watering, and rotate crops for 2-3 years away from solanaceous plants.",
    },
    "Tomato___Early_blight": {
        "display_name": "Tomato Early Blight",
        "healthy": False,
        "treatment": "Remove lower infected leaves. Apply fungicides (chlorothalonil or mancozeb) at first symptoms. Mulch to prevent soil splash and rotate crops annually.",
    },
    "Tomato___Late_blight": {
        "display_name": "Tomato Late Blight",
        "healthy": False,
        "treatment": "Act quickly with fungicides (chlorothalonil or copper-based), as this disease spreads rapidly in cool, wet weather. Destroy infected plants and avoid overhead irrigation.",
    },
    "Tomato___Leaf_Mold": {
        "display_name": "Tomato Leaf Mold",
        "healthy": False,
        "treatment": "Improve greenhouse/garden ventilation and reduce humidity. Apply fungicides (chlorothalonil or copper-based) and remove infected leaves promptly.",
    },
    "Tomato___Septoria_leaf_spot": {
        "display_name": "Tomato Septoria Leaf Spot",
        "healthy": False,
        "treatment": "Remove infected lower leaves. Apply fungicides (chlorothalonil or mancozeb), mulch around plants, and avoid overhead watering.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "display_name": "Tomato Spider Mites (Two-Spotted)",
        "healthy": False,
        "treatment": "Spray plants with water to dislodge mites, or apply insecticidal soap / neem oil. Introduce predatory mites for biological control in persistent infestations.",
    },
    "Tomato___Target_Spot": {
        "display_name": "Tomato Target Spot",
        "healthy": False,
        "treatment": "Apply fungicides (chlorothalonil or azoxystrobin) at early symptoms. Improve air circulation through staking/pruning and avoid overhead irrigation.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "display_name": "Tomato Yellow Leaf Curl Virus",
        "healthy": False,
        "treatment": "No cure exists; remove and destroy infected plants. Control whitefly vectors with insecticides or reflective mulches, and use resistant tomato varieties.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "display_name": "Tomato Mosaic Virus",
        "healthy": False,
        "treatment": "No cure exists; remove and destroy infected plants. Disinfect tools between plants, wash hands after handling tobacco products, and use resistant varieties.",
    },
    "Tomato___healthy": {
        "display_name": "Healthy Tomato",
        "healthy": True,
        "treatment": "No treatment needed. Maintain consistent watering, staking, and regular pest monitoring.",
    },
}


def get_disease_info(class_name: str) -> dict:
    """Safe lookup with a sensible fallback for unknown class names."""
    return DISEASE_INFO.get(
        class_name,
        {
            "display_name": class_name.replace("___", " - ").replace("_", " "),
            "healthy": "healthy" in class_name.lower(),
            "treatment": "No specific treatment info available for this class.",
        },
    )
