from transformers import pipeline
from PIL import Image

# Load model once (global)
classifier = pipeline(
    "image-classification",
    model="dima806/chest_xray_pneumonia_detection"
)

def analyze_xray(image_file):
    image = Image.open(image_file)

    result = classifier(image)

    label = result[0]["label"]
    score = result[0]["score"]

    return label, score