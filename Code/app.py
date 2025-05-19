import streamlit as st
import numpy as np
import cv2
import plotly.express as px
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Dummy login credentials
USERS = {"admin": "password123", "user": "test123"}

# Load the trained model
model = load_model('deepfake_detection_model.h5')

def preprocess_image(image):
    image = cv2.resize(image, (96, 96))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

def predict_image(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    class_label = np.argmax(prediction, axis=1)[0]
    return "Fake" if class_label == 0 else "Real"

def login():
    st.markdown("""
        <style>
            .main { background-color: #f5f5f5; }
        </style>
    """, unsafe_allow_html=True)
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USERS and USERS[username] == password:
            st.session_state["logged_in"] = True
        else:
            st.error("Invalid username or password")

def home():
    st.markdown("<h1 style='text-align: center; color: white;'>🎭 DEEP FAKE DETECTION</h1>", unsafe_allow_html=True)
    st.image("coverpage.png", width=650)  # You can adjust width as needed


def deepfake_understanding():
    st.markdown("<h2 style='font-size:30px;'>📘 Understanding Deepfakes</h2>", unsafe_allow_html=True)
    st.write("""
    Deepfakes are synthetic media where a person in an existing image or video is replaced with someone else's likeness. Leveraging sophisticated AI algorithms, primarily deep learning techniques, deepfakes can create incredibly realistic and convincing fake videos and images. 

    While they have legitimate uses in entertainment and education, deepfakes pose significant ethical and security challenges. They can be used to spread misinformation, create malicious content, and impersonate individuals without consent.

    Detecting deepfakes is crucial to ensure digital media integrity. AI models can analyze subtle artifacts and inconsistencies that humans might miss, allowing us to determine whether an image or video is fake or real.
    """)

def predict_page():
    st.markdown("<h2 style='font-size:30px;'>🖼️ Upload Image for Prediction</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, channels="BGR", caption="Uploaded Image")
        result = predict_image(image)

        if result == "Fake":
            color = "red"
            description = "The image appears to be a deepfake based on model analysis of inconsistencies."
        else:
            color = "green"
            description = "The image appears to be real without signs of manipulation."

        st.markdown(f"<h2 style='color:{color}; font-size:28px;'>Prediction: {result}</h2>", unsafe_allow_html=True)
        st.write(description)

def graph_page():
    st.markdown("<h2 style='font-size:30px;'>📈 Model Performance</h2>", unsafe_allow_html=True)

    # Static image backup
    st.markdown("### Training Accuracy and Loss")
    st.image("Figure_2.png")
    st.image("Figure_1.png")

    # Interactive chart example
    st.markdown("### 📊 Interactive Accuracy Graph")
    data = pd.DataFrame({
        "Epoch": list(range(1, 11)),
        "Accuracy": [0.60, 0.72, 0.78, 0.81, 0.85, 0.87, 0.90, 0.92, 0.94, 0.95],
        "Loss": [0.9, 0.7, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]
    })
    fig = px.line(data, x="Epoch", y=["Accuracy", "Loss"], markers=True)
    fig.update_layout(title="Model Training Metrics", xaxis_title="Epoch", yaxis_title="Value")
    st.plotly_chart(fig)

def main():
    st.set_page_config(page_title="Deepfake Detection Dashboard", layout="wide")
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        login()
    else:
        st.sidebar.markdown("""
            <style>
                .css-1d391kg, .css-1v3fvcr, .css-h5rgaw, .css-18e3th9 {
                    background-color: #31333F;
                    color: white;
                    font-size: 18px;
                    border-radius: 10px;
                }
            </style>
        """, unsafe_allow_html=True)

        st.sidebar.markdown("<h2 style='color: cyan;'>🧭 Navigation</h2>", unsafe_allow_html=True)
        page = st.sidebar.radio("", ["🏠 Home", "📘 Deepfake Understanding", "🖼️ Predict Image", "📈 Graphs"])

        if page == "🏠 Home":
            home()
        elif page == "📘 Deepfake Understanding":
            deepfake_understanding()
        elif page == "🖼️ Predict Image":
            predict_page()
        elif page == "📈 Graphs":
            graph_page()

        if st.sidebar.button("🚪 Logout"):
            st.session_state["logged_in"] = False
            st.experimental_rerun()

if __name__ == "__main__":
    main()
