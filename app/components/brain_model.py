from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# 🔥 load model once
model = load_model("brain_tumor_model.h5")

def analyze_brain(uploaded_file):
    # convert image
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224,224))

    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # prediction
    pred = model.predict(image)[0][0]

    if pred > 0.5:
        return "TUMOR", float(pred)
    else:
        return "NO_TUMOR", float(1 - pred)