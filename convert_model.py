import tensorflow as tf

print("Loading model...")

model = tf.keras.models.load_model(
    "waste_classifier_model.h5",
    compile=False
)

print("Saving model...")

model.save("waste_classifier_model.keras")

print("SUCCESS!")