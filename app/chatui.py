import streamlit as st
import requests
from PIL import Image
import io
import json
import openai
import tempfile
import cv2
import numpy as np
from ultralytics import YOLO
from flask import Flask, request, jsonify
import os

# Set up Flask app
app = Flask(__name__)

# Load YOLO models
model_anomaly = YOLO("anamoly.pt")  # Anomaly detection model
model_cavity = YOLO("hi.pt")  # Cavity detection model

# OpenAI API Key (Ensure it's set in the environment)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key is missing. Set it as an environment variable.")
openai.api_key = OPENAI_API_KEY

def detect_disease(image):
    """Processes the uploaded image and runs YOLO detection models."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        image.save(temp_file, format='JPEG')
        temp_filepath = temp_file.name

    results_anomaly = model_anomaly.predict(temp_filepath)
    results_cavity = model_cavity.predict(temp_filepath)

    condition = "No significant condition detected."
    if results_anomaly and results_anomaly[0].boxes:
        condition = "Possible Medical Anomaly detected."
    elif results_cavity and results_cavity[0].boxes:
        condition = "Possible Dental Cavity detected."

    return condition

def chatbot_reply(query):
    """Fetches response from OpenAI ChatGPT model."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": query}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error fetching response: {str(e)}"

@app.route("/chat", methods=["POST"])
def chat():
    """Handles chatbot queries via API."""
    data = request.json
    user_query = data.get("query", "")
    response = chatbot_reply(user_query)
    return jsonify({"reply": response})

@app.route("/upload", methods=["POST"])
def upload_image():
    """Handles image uploads and processes them using YOLO models."""
    file = request.files["file"]
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    image = Image.open(file)
    result = detect_disease(image)
    return jsonify({"analysis": result})

# Streamlit UI
st.set_page_config(page_title="AI Medical Chatbot", layout="wide")
st.title("🩺 AI-Powered Medical Diagnosis Chatbot")
st.write("Upload a medical image or video for analysis, take a photo, or ask a medical query.")

col1, col2 = st.columns([2, 1])
with col1:
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png", "jpeg", "mp4", "avi"],
                                     accept_multiple_files=False)
    st.write("Or drag and drop an image or video file here.")
    image_capture = st.camera_input("Take a photo")

    image = None
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    if image_capture:
        image = Image.open(image_capture)
        st.image(image, caption="Captured Image", use_container_width=True)

    if image:
        if st.button("Analyze Image"):
            detected_condition = detect_disease(image)
            st.subheader(detected_condition)

with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/4713/4713869.png", width=200)
    st.write("🤖 AI Chatbot Assistant")

    user_query = st.text_area("Ask a medical question or enter a drug name:")
    submit_button = st.button("Enter")

    if submit_button and user_query:
        response = chatbot_reply(user_query)
        st.subheader("Response:")
        st.write(response)

if __name__ == "__main__":
    app.run(port=5000, debug=True)