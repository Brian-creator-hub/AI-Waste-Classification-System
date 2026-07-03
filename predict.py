import cv2
import numpy as np
from tensorflow.keras.models import load_model


model = load_model("waste_classifier_model.h5")

categories = [
    "Hazardous",
    "Non-Recyclable",
    "Organic",
    "Recyclable"
]


camera = cv2.VideoCapture(0)


while True:

    ret, frame = camera.read()

    if not ret:
        break


    # IMPORTANT: same size as training
    img = cv2.resize(frame, (128,128))

    img = img.astype("float32") / 255.0

    img = np.expand_dims(img, axis=0)


    prediction = model.predict(img, verbose=0)

    confidence = np.max(prediction) * 100

    predicted_class = categories[np.argmax(prediction)]


    text = f"{predicted_class}: {confidence:.2f}%"


    cv2.putText(
        frame,
        text,
        (20,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )


    cv2.imshow("AI Waste Sorter", frame)


    if cv2.waitKey(1) == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()