import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(
    page_title="AI Waste Classification System",
    page_icon="♻️",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------
st.markdown("""
<style>

.main{
    background-color:#f5f7fa;
}

.title{
    text-align:center;
    color:#2E8B57;
    font-size:42px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
}

.prediction-box{
    background:#e8f5e9;
    border-radius:15px;
    padding:20px;
    text-align:center;
    box-shadow:2px 2px 8px rgba(0,0,0,0.2);
}

.footer{
    text-align:center;
    color:gray;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "waste_classifier_model.keras",
        compile=False
    )

model = load_model()

# -------------------------------------------------
# CLASS NAMES
# -------------------------------------------------
classes = [
    "Hazardous",
    "Non-Recyclable",
    "Organic",
    "Recyclable"
]

icons = {
    "Hazardous":"☣️",
    "Non-Recyclable":"🗑️",
    "Organic":"🌿",
    "Recyclable":"♻️"
}

tips = {

"Hazardous":
"""
Dispose only at authorized hazardous waste collection centres.

Examples:

• Batteries

• Chemicals

• Paint

• Medical waste
""",

"Non-Recyclable":
"""
Dispose responsibly using normal waste bins.

Examples:

• Dirty plastics

• Diapers

• Styrofoam
""",

"Organic":
"""
Organic waste can be composted.

Examples:

• Food leftovers

• Fruits

• Vegetables

• Leaves
""",

"Recyclable":
"""
Place inside recycling bins.

Examples:

• Plastic bottles

• Paper

• Glass

• Metal cans
"""

}

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.title("♻️ AI Waste Classifier")

st.sidebar.success("Model")

st.sidebar.write("MobileNetV2")

st.sidebar.markdown("---")

st.sidebar.subheader("Waste Categories")

st.sidebar.write("🌿 Organic")

st.sidebar.write("♻️ Recyclable")

st.sidebar.write("🗑️ Non-Recyclable")

st.sidebar.write("☣️ Hazardous")

st.sidebar.markdown("---")

st.sidebar.info(
"""
Upload any waste image and the AI
will predict its category.
"""
)

# -------------------------------------------------
# TITLE
# -------------------------------------------------

st.markdown(
'<p class="title">♻️ AI Waste Classification System</p>',
unsafe_allow_html=True
)

st.markdown(
'<p class="subtitle">Upload an image to classify waste using Artificial Intelligence</p>',
unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
"Upload Waste Image",
type=["jpg","jpeg","png"]
)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1,col2 = st.columns(2)

    with col1:

       st.image(
    image,
    caption="Uploaded Image",
    width=400
)
    img=np.array(image)

    img=cv2.resize(img,(128,128))

    img=img.astype("float32")/255.0

    img=np.expand_dims(img,axis=0)

    with st.spinner("AI is analyzing image..."):

        prediction=model.predict(img)

    predicted=np.argmax(prediction)

    confidence=float(np.max(prediction)*100)

    waste=classes[predicted]

    with col2:

        st.markdown(
        f"""
        <div class="prediction-box">

        <h1>{icons[waste]}</h1>

        <h2>{waste}</h2>

        </div>

        """,
        unsafe_allow_html=True
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(confidence/100)

        if confidence>=90:

            st.success("Very High Confidence")

        elif confidence>=70:

            st.info("High Confidence")

        elif confidence>=50:

            st.warning("Moderate Confidence")

        else:

            st.error("Low Confidence")

    st.markdown("---")

    st.subheader("Prediction Probabilities")

    for i,name in enumerate(classes):

        st.write(name)

        st.progress(float(prediction[0][i]))

        st.write(f"{prediction[0][i]*100:.2f}%")

    st.markdown("---")

    st.subheader("♻️ Disposal Recommendation")

    st.info(tips[waste])

# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.markdown("---")

st.markdown(
"""
### Technologies Used

- TensorFlow
- MobileNetV2
- Streamlit
- OpenCV
- NumPy
- Pillow

---

Developed by **Brian Berur**

Mount Kenya University

2026
"""
)