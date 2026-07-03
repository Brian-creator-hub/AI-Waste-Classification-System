from tensorflow.keras.models import load_model

print("Loading model...")

model = load_model("waste_classifier_model.h5")

print("SUCCESS!")
print("Model loaded correctly.")