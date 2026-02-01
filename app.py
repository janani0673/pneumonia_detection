import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
from datetime import date, datetime

# 🩺 Streamlit Page Config
st.set_page_config(page_title="Pneumonia Detection", page_icon="🩺", layout="wide")

# 🌟 Background Styling
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://user-images.githubusercontent.com/76659596/107152793-30234600-696a-11eb-8827-56cb0c3a7578.png');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        min-height: 100vh;
    }

    .content-container {
        background-color: rgba(0, 0, 0, 0.5);
        padding: 20px;
        border-radius: 12px;
    }

    h1 { color: #FFD700; text-shadow: 2px 2px 5px #000000; }
    h2, h3 { color: #00CED1; text-shadow: 1px 1px 3px #000000; }
    p, span { color: #FFFFFF; text-shadow: 1px 1px 2px #000000; }

    .stButton>button {
        background-color: #FF6347;
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 8px 24px;
        font-size: 16px;
    }

    div.stFileUploader {
        background-color: rgba(255, 255, 255, 0.85);
        border-radius: 12px;
        padding: 10px;
    }

    .stInfo { background-color: rgba(0, 128, 128, 0.8) !important; color: white; font-weight: bold; }
    .stSuccess { background-color: rgba(50, 205, 50, 0.8) !important; color: white; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True
)

# Wrap content in a semi-transparent container
st.markdown('<div class="content-container">', unsafe_allow_html=True)

# 🌟 Load Model
@st.cache_resource
def load_pneumonia_model():
    model = tf.keras.models.load_model("pneumonia_model.h5")
    return model

model = load_pneumonia_model()

# 📋 App Title
st.title("🩻 Pneumonia Detection from Chest X-Ray")
st.markdown(
    """
    <p style="color:black; font-size:18px; text-shadow: 1px 1px 3px #000000;">
    Upload a chest X-ray image to detect if pneumonia is present. 
    The AI model will analyze the image and provide a diagnosis.
    </p>
    """,
    unsafe_allow_html=True
)

# 🧍 Patient Information
st.subheader("👩‍⚕️ Patient Details")
patient_name = st.text_input("Patient Name:")

dob_date = st.date_input("Date of Birth:")
dob_time = st.time_input("Time of Birth:", value=datetime.now().time())
dob = datetime.combine(dob_date, dob_time)

# 🖼️ File Uploader
uploaded_file = st.file_uploader("Upload Chest X-Ray Image", type=["jpg", "jpeg", "png"])

# 🧠 Prediction Function
def predict_pneumonia(image_path):
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction[0][0]

# 📑 PDF Report using BytesIO
def generate_report_bytes(name, dob, result, diagnosis):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(300, 750, "Pneumonia Detection Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, 700, f"Patient Name: {name}")
    c.drawString(50, 680, f"Date of Birth: {dob.strftime('%d-%m-%Y %H:%M:%S')}")
    c.drawString(50, 660, f"Diagnosis Date: {date.today().strftime('%d-%m-%Y')}")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 620, f"Prediction Result: {result}")
    c.setFont("Helvetica", 12)
    c.drawString(50, 590, f"Diagnosis: {diagnosis}")

    c.save()
    buffer.seek(0)
    return buffer

# 🚀 Prediction and Report
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Chest X-Ray", use_column_width=True)
    st.write("Analyzing the image... Please wait ⏳")

    # Save temporarily for model
    with BytesIO(uploaded_file.read()) as tmp:
        pred = predict_pneumonia(tmp)
    
    if pred > 0.5:
        result = "🩺 Pneumonia Detected"
        diagnosis = "Signs of lung infection are visible. Consult a healthcare provider."
    else:
        result = "✅ Normal - No Pneumonia Detected"
        diagnosis = "Lungs appear clear. Maintain healthy habits."

    # Display results
    st.subheader("Result:")
    st.success(result)
    st.info(f"Model Confidence: {pred*100:.2f}%")
    st.write(f"**Diagnosis Suggestion:** {diagnosis}")

    # PDF download
    pdf_bytes = generate_report_bytes(patient_name, dob, result, diagnosis)
    st.download_button(
        label="📥 Download Report (PDF)",
        data=pdf_bytes,
        file_name=f"{patient_name}_Pneumonia_Report.pdf",
        mime="application/pdf"
    )

# Footer
st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")
st.caption("AI Pneumonia Detection Web App")
