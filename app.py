
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Waste Classification System",
    page_icon="♻️",
    layout="wide"
)

# --------------------------------------------------
# WASTE CATEGORIES
# Keep this order consistent with your training data.
# --------------------------------------------------

CLASS_NAMES = [
    "Hazardous",
    "Non-Recyclable",
    "Organic",
    "Recyclable"
]

DISPOSAL_GUIDANCE = {
    "Hazardous": (
        "Handle with care. Keep this waste separate from ordinary rubbish. "
        "Use an appropriate hazardous-waste collection or disposal service."
    ),
    "Non-Recyclable": (
        "Place this waste in the appropriate general-waste bin, following "
        "your local waste collection guidelines."
    ),
    "Organic": (
        "Where facilities are available, place this waste in an organic-waste "
        "bin or use it for composting."
    ),
    "Recyclable": (
        "Keep this item separate from general waste. If appropriate, clean it "
        "and place it in a recycling collection bin."
    )
}

MODEL_PATH = Path(__file__).parent / "waste_classifier_model.keras"

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_waste_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Place the model file in the same folder as app.py."
        )

    return tf.keras.models.load_model(MODEL_PATH)


# --------------------------------------------------
# IMAGE PREPROCESSING
# Images are resized to 128 x 128 and normalized to [0, 1].
# This matches the preprocessing described for your model.
# --------------------------------------------------

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((128, 128))

    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# --------------------------------------------------
# APPLICATION HEADER
# --------------------------------------------------

st.title("♻️ AI-Based Waste Classification System")

st.write(
    "This system uses a trained deep-learning model to classify waste "
    "images into four categories and provide disposal guidance."
)

st.info(
    "For the best result, use a clear picture showing one waste item "
    "with good lighting."
)

st.divider()

# --------------------------------------------------
# IMAGE SOURCE SELECTION
# --------------------------------------------------

st.subheader("Choose an Image")

image_source = st.radio(
    "How would you like to provide the waste image?",
    ["Upload Waste Image", "Take a Picture"],
    horizontal=True
)

uploaded_file = None

if image_source == "Upload Waste Image":
    uploaded_file = st.file_uploader(
        "Choose an image from your device",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG."
    )

else:
    uploaded_file = st.camera_input(
        "Take a picture of the waste",
        help="Allow camera access if your browser requests permission."
    )

# --------------------------------------------------
# CLASSIFICATION
# --------------------------------------------------

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Selected Waste Image")

        st.image(
            image,
            caption="Image provided for classification",
            use_column_width=True
        )

        if st.button("Classify Waste", type="primary"):

            with st.spinner("Analysing image... Please wait."):

                model = load_waste_model()

                processed_image = preprocess_image(image)

                predictions = model.predict(
                    processed_image,
                    verbose=0
                )

                probabilities = np.asarray(predictions)[0].reshape(-1)

                if len(probabilities) != len(CLASS_NAMES):
                    st.error(
                        "The model output does not contain four classes. "
                        "Check your model's output layer and the class order "
                        "used during training."
                    )
                    st.stop()

                predicted_index = int(np.argmax(probabilities))
                predicted_class = CLASS_NAMES[predicted_index]
                confidence = float(probabilities[predicted_index]) * 100

            # ------------------------------------------
            # DISPLAY PREDICTION
            # ------------------------------------------

            st.divider()
            st.subheader("Classification Result")

            result_col, confidence_col = st.columns(2)

            with result_col:
                st.metric(
                    label="Predicted Waste Category",
                    value=predicted_class
                )

            with confidence_col:
                st.metric(
                    label="Model Confidence",
                    value=f"{confidence:.2f}%"
                )

            st.subheader("Disposal Recommendation")

            st.success(DISPOSAL_GUIDANCE[predicted_class])

            # ------------------------------------------
            # DISPLAY CLASS PROBABILITIES
            # ------------------------------------------

            st.subheader("Prediction Probabilities")

            probability_data = pd.DataFrame({
                "Waste Category": CLASS_NAMES,
                "Probability (%)": probabilities * 100
            })

            probability_data = probability_data.sort_values(
                by="Probability (%)",
                ascending=False
            )

            st.bar_chart(
                probability_data.set_index("Waste Category")
            )

            st.dataframe(
                probability_data.style.format({
                    "Probability (%)": "{:.2f}%"
                }),
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Confidence is the model's prediction score, not a guarantee "
                "that the classification is correct. Verify hazardous items "
                "carefully and follow local disposal regulations."
            )

    except FileNotFoundError as error:
        st.error(str(error))

    except Exception as error:
        st.error(
            "The image could not be classified. Check that the model file "
            "is valid and that the required Python packages are installed."
        )
        st.exception(error)

else:
    st.info(
        "Upload an image or take a picture to begin waste classification."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "AI-Based Waste Classification System | "
    "Deep Learning Project"
)
