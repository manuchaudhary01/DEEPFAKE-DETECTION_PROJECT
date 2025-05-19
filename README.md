# Deepfake Detection Web Application

A web-based application to detect deepfake videos using a deep learning model. This project uses a Streamlit front end and a Flask API backend to analyze uploaded video files and predict whether they are real or fake.

---

## 🔍 Features

- Upload and analyze video files for deepfake content
- Uses MobileNetV2-based deep learning model
- Streamlit front-end interface
- Real-time feedback with prediction results
- Logging and error handling

---

## 🛠️ Tech Stack

- Python
- Streamlit (Frontend)
- Flask (Backend/API)
- TensorFlow / Keras
- OpenCV
- NumPy, Pandas
- scikit-learn
- Gunicorn (for deployment, optional)

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/deepfake-detection-app.git
cd deepfake-detection-app

Create and Activate Virtual Environment
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows

Install Dependencies
pip install -r requirements.txt

Open a new terminal and run:
cd code 
streamlit run app.py