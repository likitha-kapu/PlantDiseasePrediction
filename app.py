# Inside app.py

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

MODEL_PATH = "model/best_model.h5"
LABELS_PATH = "model/labels.txt"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load labels
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]

# ✅ Corrected disease_info dictionary (keys match labels.txt exactly)
disease_info = {
    "Apple___Apple_scab": {
        "description": "Apple scab is a fungal disease caused by Venturia inaequalis. It causes dark lesions on leaves and fruits.",
        "treatment": "Prune affected parts, apply fungicides like captan or mancozeb, and use resistant apple varieties."
    },
    "Apple___Black_rot": {
        "description": "Black rot is a fungal disease caused by Botryosphaeria obtusa. It affects fruit, leaves, and twigs.",
        "treatment": "Remove mummified fruits, prune infected twigs, and apply fungicides such as thiophanate-methyl."
    },
    "Apple___Cedar_apple_rust": {
        "description": "Cedar apple rust is caused by the fungus Gymnosporangium juniperi-virginianae. It requires both apple and cedar trees to complete its life cycle.",
        "treatment": "Remove nearby cedar trees if possible, apply fungicides like myclobutanil, and use resistant varieties."
    },
    "Apple___healthy": {
        "description": "The leaf is healthy with no visible signs of disease.",
        "treatment": "No treatment needed."
    },
    "Blueberry___healthy": {
        "description": "The blueberry leaf is healthy with no disease symptoms.",
        "treatment": "No treatment needed."
    },
    "Cherry___Powdery_mildew": {
        "description": "Powdery mildew is caused by Podosphaera clandestina. It forms white powdery spots on leaves.",
        "treatment": "Apply sulfur-based fungicides, prune infected parts, and improve air circulation."
    },
    "Cherry___healthy": {
        "description": "The cherry leaf is healthy with no visible symptoms.",
        "treatment": "No treatment needed."
    },
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": {
        "description": "Gray leaf spot is caused by Cercospora zeae-maydis. It produces rectangular lesions on corn leaves.",
        "treatment": "Rotate crops, use resistant hybrids, and apply fungicides like strobilurins or triazoles."
    },
    "Corn___Common_rust": {
        "description": "Common rust is caused by Puccinia sorghi. It produces reddish-brown pustules on leaves.",
        "treatment": "Use resistant hybrids and apply fungicides like mancozeb if severe."
    },
    "Corn___Northern_Leaf_Blight": {
        "description": "Northern leaf blight is caused by Exserohilum turcicum. It causes large cigar-shaped lesions.",
        "treatment": "Use resistant hybrids, rotate crops, and apply fungicides when needed."
    },
    "Corn___healthy": {
        "description": "The corn leaf is healthy with no visible symptoms.",
        "treatment": "No treatment needed."
    },
    "Grape___Black_rot": {
        "description": "Black rot of grapes is caused by Guignardia bidwellii. It causes circular lesions on leaves and rots fruit.",
        "treatment": "Apply fungicides like mancozeb or copper sprays and remove infected parts."
    },
    "Grape___Esca_(Black_Measles)": {
        "description": "Esca is a fungal disease complex that causes black streaks and tiger-stripe leaf patterns.",
        "treatment": "Prune infected wood, avoid water stress, and apply fungicides where effective."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "description": "Leaf blight caused by Pseudocercospora causes brown angular leaf spots.",
        "treatment": "Apply fungicides such as copper oxychloride and practice good vineyard sanitation."
    },
    "Grape___healthy": {
        "description": "The grape leaf is healthy with no visible symptoms.",
        "treatment": "No treatment needed."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "description": "Citrus greening is a bacterial disease caused by Candidatus Liberibacter spp. It causes yellow shoots and bitter fruits.",
        "treatment": "No cure exists. Control psyllid vectors, remove infected trees, and plant disease-free stock."
    },
    "Peach___Bacterial_spot": {
        "description": "Bacterial spot is caused by Xanthomonas campestris. It causes small dark lesions on leaves and fruits.",
        "treatment": "Use copper-based bactericides, prune infected parts, and plant resistant cultivars."
    },
    "Peach___healthy": {
        "description": "The peach leaf is healthy with no visible symptoms.",
        "treatment": "No treatment needed."
    },
    "Pepper__bell___Bacterial_spot": {
        "description": "Bacterial spot is caused by Xanthomonas campestris. It leads to water-soaked lesions.",
        "treatment": "Use copper sprays, resistant cultivars, and rotate crops."
    },
    "Pepper__bell___healthy": {
        "description": "The bell pepper leaf is healthy.",
        "treatment": "No treatment needed."
    },
    "Potato___Early_blight": {
        "description": "Early blight is caused by Alternaria solani. It produces concentric ring lesions on leaves.",
        "treatment": "Use fungicides like chlorothalonil and rotate crops."
    },
    "Potato___Late_blight": {
        "description": "Late blight is caused by Phytophthora infestans. It causes dark lesions and rapid leaf death.",
        "treatment": "Apply fungicides like mancozeb, destroy infected plants, and avoid overhead irrigation."
    },
    "Potato___healthy": {
        "description": "The potato leaf is healthy.",
        "treatment": "No treatment needed."
    },
    "Raspberry___healthy": {
        "description": "The raspberry leaf is healthy.",
        "treatment": "No treatment needed."
    },
    "Soybean___healthy": {
        "description": "The soybean leaf is healthy.",
        "treatment": "No treatment needed."
    },
    "Squash___Powdery_mildew": {
        "description": "Powdery mildew affects squash leaves with white patches.",
        "treatment": "Apply sulfur-based fungicides and use resistant varieties."
    },
    "Strawberry___Leaf_scorch": {
        "description": "Leaf scorch causes burnt-looking brown edges on strawberry leaves.",
        "treatment": "Remove infected leaves, avoid overhead irrigation, and improve spacing."
    },
    "Strawberry___healthy": {
        "description": "The strawberry leaf is healthy.",
        "treatment": "No treatment needed."
    },
    "Tomato___Bacterial_spot": {
        "description": "Bacterial spot is caused by Xanthomonas. It creates water-soaked lesions on leaves and fruit.",
        "treatment": "Apply copper sprays, rotate crops, and plant resistant varieties."
    },
    "Tomato___Early_blight": {
        "description": "Early blight is caused by Alternaria solani. It leads to concentric ring lesions.",
        "treatment": "Spray fungicides like chlorothalonil and practice crop rotation."
    },
    "Tomato___Late_blight": {
        "description": "Late blight is caused by Phytophthora infestans. It leads to rapid leaf and fruit rot.",
        "treatment": "Apply mancozeb or copper fungicides and destroy infected plants."
    },
    "Tomato___Leaf_Mold": {
        "description": "Leaf mold is caused by Passalora fulva. It appears as yellow patches on leaves with olive-green mold.",
        "treatment": "Improve air circulation and use fungicides like chlorothalonil."
    },
    "Tomato___Septoria_leaf_spot": {
        "description": "Septoria leaf spot is caused by Septoria lycopersici. It causes circular brown spots.",
        "treatment": "Remove infected leaves and apply fungicides."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "description": "Two-spotted spider mites cause stippling and webbing on tomato leaves.",
        "treatment": "Spray with miticides or neem oil and encourage natural predators."
    },
    "Tomato___Target_Spot": {
        "description": "Target spot is caused by Corynespora cassiicola. It causes concentric lesions on tomato leaves.",
        "treatment": "Use resistant varieties and apply fungicides."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "TYLCV is a viral disease spread by whiteflies. It causes curling and yellowing of leaves.",
        "treatment": "Control whiteflies and plant resistant varieties."
    },
    "Tomato___Tomato_mosaic_virus": {
        "description": "Tomato mosaic virus causes mottling and distortion of leaves.",
        "treatment": "Remove infected plants, disinfect tools, and use resistant seeds."
    },
    "Tomato___healthy": {
        "description": "The tomato leaf is healthy.",
        "treatment": "No treatment needed."
    }
}



def model_predict(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    class_idx = np.argmax(preds[0])
    confidence = float(np.max(preds[0]) * 100)
    class_name = class_names[class_idx]

    return class_name, confidence

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    class_name, confidence = model_predict(filepath)

    # Get disease info if available
    info = disease_info.get(class_name, {
        "description": "No description available.",
        "treatment": "No treatment info available."
    })

    return jsonify({
        "class_name": class_name,
        "confidence": f"{confidence:.2f}%",
        "description": info["description"],
        "treatment": info["treatment"]
    })

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
