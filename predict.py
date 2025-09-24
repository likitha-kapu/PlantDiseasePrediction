import tensorflow as tf
import numpy as np
import cv2
import os

# Load the trained model
model = tf.keras.models.load_model("model/best_model.h5")

# Class indices mapping (label names)
# Ensure you use the same dataset path used in training
dataset_path = "plantvillage"
class_names = sorted(os.listdir(dataset_path))

def predict_image(img_path):
    # Load and preprocess image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
    img = cv2.resize(img, (224, 224))  # resize to MobileNetV2 input
    img = img / 255.0  # normalize
    img = np.expand_dims(img, axis=0)  # add batch dimension

    # Predict
    preds = model.predict(img)
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)

    return class_names[pred_class], confidence

# Test with a sample image
image_path = "sample_leaf.jpg"  # replace with your image path
if os.path.exists(image_path):
    predicted_class, confidence = predict_image(image_path)
    print(f"✅ Predicted: {predicted_class} ({confidence*100:.2f}%)")
else:
    print("⚠️ Please place a sample image in the project folder and update 'image_path'.")
