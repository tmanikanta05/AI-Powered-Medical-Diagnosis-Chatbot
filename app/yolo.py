import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageDraw
import tempfile
from ultralytics import YOLO

# 🔹 Configure Gemini AI
genai.configure(api_key="AIzaSyDJcrPX-cx7z6gAsBssQUgfdX60RmMEnpg")
GEMINI_MODEL = "gemini-1.5-flash"

# 🔹 Load YOLO models
models = {
    "Anomaly": YOLO("anamoly.pt"),
    "Cavity": YOLO("hi.pt"),
    "Fracture": YOLO("fracture.pt"),
    "Eye Disease": YOLO("eyeDisease.pt"),
    "Monkeypox": YOLO("Monkeypox.pt"),
    "Liver Disease": YOLO("liverDisease.pt"),
  #  "Brain Tumor": YOLO("brain_tumor.pt"),
}


# 🔹 Disease detection function (with bounding boxes)
def detect_disease(image):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        image.save(temp_file, format='JPEG')
        temp_filepath = temp_file.name

    detected_conditions = []
    detected_boxes = []

    for disease, model in models.items():
        results = model.predict(temp_filepath)
        if results and len(results[0].boxes) > 0:
            detected_disease = set()
            for box in results[0].boxes:
                class_id = int(box.cls.item())  # Get class ID
                confidence = round(box.conf.item() * 100, 2)  # Confidence Score
                detected_disease.add(f"{results[0].names[class_id]} ({confidence}%)")

                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers
                detected_boxes.append((x1, y1, x2, y2, results[0].names[class_id]))

            if detected_disease:
                detected_conditions.append(f"{disease} ({', '.join(detected_disease)})")

    return detected_conditions, detected_boxes  # Returns both disease names and bounding boxes


# 🔹 Draw bounding boxes on image
def draw_bounding_boxes(image, detected_boxes):
    draw = ImageDraw.Draw(image)
    for box in detected_boxes:
        x1, y1, x2, y2, label = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 10), label, fill="red")  # Add label
    return image


# 🔹 Fetch medical insights using Gemini API
def get_medical_insights(query):
    try:
        gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        response = gemini_model.generate_content(query)
        return response.text if response.text else "No insights available."
    except Exception as e:
        return f"⚠️ AI Error: {str(e)}"


# 🔹 Streamlit UI
st.set_page_config(page_title="AI Medical Chatbot", layout="wide")
st.title("🩺 AI-Powered Medical Diagnosis & Chatbot")

# 📌 Tabs for Image Analysis & Chatbot
tab1, tab2 = st.tabs(["📷 Medical Image Analysis", "💬 AI Chatbot"])

# 📷 **Tab 1: Medical Image Analysis**
with tab1:
    st.write("Upload an image or take a photo to detect conditions.")

    # 📷 Toggle Camera
    if "camera_active" not in st.session_state:
        st.session_state["camera_active"] = False  # Default: Camera OFF

    if st.button("📷 Open Camera"):
        st.session_state["camera_active"] = not st.session_state["camera_active"]  # Toggle state

    image = None
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    # Show camera input **only if the button was pressed**
    if st.session_state["camera_active"]:
        image_capture = st.camera_input("Take a photo")
        if image_capture:
            image = Image.open(image_capture)

    if uploaded_file:
        image = Image.open(uploaded_file)

    if image:
        st.image(image, caption="Selected Image", use_container_width=True)

        if st.button("🔍 Analyze Image"):
            detected_conditions, detected_boxes = detect_disease(image)

            if detected_conditions:
                st.subheader("🩺 Detected Conditions:")
                for condition in detected_conditions:
                    st.write(f"- {condition}")

                # Draw and show bounding boxes
                image_with_boxes = draw_bounding_boxes(image.copy(), detected_boxes)
                st.image(image_with_boxes, caption="Detected Areas", use_container_width=True)

                # Get AI insights
                insights = get_medical_insights(", ".join(detected_conditions))
                st.write("### 💡 AI Insights:")
                st.write(insights)
            else:
                st.write("No significant condition detected.")

# 💬 **Tab 2: AI Chatbot**
with tab2:
    st.write("💡 Ask the AI chatbot any medical-related question.")

    # Store chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display previous messages
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_input = st.chat_input("Ask a medical question...")

    if user_input:
        # Display user message
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get AI response
        response = get_medical_insights(user_input)

        # Display AI response
        with st.chat_message("assistant"):
            st.markdown(response)

        # Store AI response
        st.session_state["messages"].append({"role": "assistant", "content": response})
