import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("models/cnn_model.h5")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(50, 50))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]
    label = "IDC Positive" if pred > 0.5 else "IDC Negative"
    confidence = round(pred if pred > 0.5 else 1 - pred, 4)

    return label, confidence

if __name__ == "__main__":
    img_path = "data/test/sample_patch.png"
    result, score = predict_image(img_path)
    print(f"{img_path} → {result} (Confidence: {score})")