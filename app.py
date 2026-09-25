import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="AI Traffic Sign Recognition",
    page_icon="🚦",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("traffic_sign_cnn(2).keras")

model = load_model()

# Standard GTSRB class names (0–42)
sign_names = {
    0: "Speed Limit 20 km/h",
    1: "Speed Limit 30 km/h",
    2: "Speed Limit 50 km/h",
    3: "Speed Limit 60 km/h",
    4: "Speed Limit 70 km/h",
    5: "Speed Limit 80 km/h",
    6: "End of Speed Limit 80 km/h",
    7: "Speed Limit 100 km/h",
    8: "Speed Limit 120 km/h",
    9: "No Passing",
    10: "No Passing for Vehicles over 3.5 Tons",
    11: "Right-of-Way at Next Intersection",
    12: "Priority Road",
    13: "Yield",
    14: "Stop",
    15: "No Vehicles",
    16: "Vehicles over 3.5 Tons Prohibited",
    17: "No Entry",
    18: "General Caution",
    19: "Dangerous Curve Left",
    20: "Dangerous Curve Right",
    21: "Double Curve",
    22: "Bumpy Road",
    23: "Slippery Road",
    24: "Road Narrows on the Right",
    25: "Road Work",
    26: "Traffic Signals",
    27: "Pedestrians",
    28: "Children Crossing",
    29: "Bicycles Crossing",
    30: "Beware of Ice/Snow",
    31: "Wild Animals Crossing",
    32: "End of All Speed and Passing Limits",
    33: "Turn Right Ahead",
    34: "Turn Left Ahead",
    35: "Ahead Only",
    36: "Go Straight or Right",
    37: "Go Straight or Left",
    38: "Keep Right",
    39: "Keep Left",
    40: "Roundabout Mandatory",
    41: "End of No Passing",
    42: "End of No Passing by Vehicles over 3.5 Tons"
}

uploaded = st.file_uploader(
    "Upload a traffic sign image",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Traffic Sign", use_container_width=True)

    if st.button("🔍 Recognize Traffic Sign", type="primary"):
        img = image.resize((32, 32))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)

        prediction = model.predict(arr, verbose=0)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)

        name = sign_names.get(predicted_class, f"Class {predicted_class}")

        st.success(f"🚦 Sign: {name}")
        st.write(f"**Confidence:** {confidence:.2f}%")

        st.subheader("📌 Meaning")
        st.write(f"This image was classified as **{name}**.")

        st.subheader("⚠️ Safety Alert")
        st.write("Follow the traffic rule indicated by the recognized sign and drive carefully.")

st.caption("AI-Based Traffic Sign Recognition Using CNN • GTSRB dataset")
