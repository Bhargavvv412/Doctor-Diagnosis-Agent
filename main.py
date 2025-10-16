import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import numpy as np
import cv2
import base64
from datetime import datetime
import json
import uuid
import os

# ---------- Helper functions ----------
def pil_to_numpy(img):
    arr = np.array(img)
    if arr.ndim == 2:
        return arr
    if arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return arr

def generate_heatmap(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap, 0.5, image_array, 0.5, 0)
    return Image.fromarray(overlay)

def get_gemini_analysis(pil_img, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = (
        "You are an expert radiologist. Analyze this chest X-ray image carefully. "
        "List radiological findings, impression, and possible diseases (ranked). "
        "Mention uncertainty and recommend follow-up if needed. Be concise."
    )
    result = model.generate_content([prompt, pil_img])
    return result.text

def save_analysis(filename, text, keywords):
    data = {
        "id": str(uuid.uuid4()),
        "file": filename,
        "datetime": datetime.now().isoformat(),
        "findings": text,
        "keywords": keywords
    }
    os.makedirs("results", exist_ok=True)
    with open(f"results/{data['id']}.json", "w") as f:
        json.dump(data, f, indent=2)
    return data

def extract_keywords(text):
    base = ["pneumonia","effusion","opacity","atelectasis","nodule","fibrosis","consolidation","mass"]
    return [k for k in base if k in text.lower()]

# ---------- Streamlit UI ----------
st.set_page_config(page_title="🩻 X-Ray Detection Agent (Gemini Free)", layout="wide")
st.title("🩻 AI X-Ray Disease Detection (Free with Gemini)")
st.write("Upload a chest X-ray and let Google Gemini analyze it — **free & vision-powered** 💫")

api_key = st.text_input("🔑 Enter your Gemini API Key (get from https://aistudio.google.com/)", type="password")
if not api_key:
    st.warning("Please enter your Gemini API key above.")
    st.stop()

uploaded_file = st.file_uploader("📤 Upload Chest X-Ray (JPG / PNG)", type=["jpg", "jpeg", "png"])
if not uploaded_file:
    st.info("Upload an X-ray image to start analysis.")
    st.stop()

# Display uploaded image
img = Image.open(uploaded_file).convert("RGB")
st.image(img, caption="Uploaded X-ray", use_column_width=True)

# Generate and show heatmap
overlay = generate_heatmap(pil_to_numpy(img))
st.image(overlay, caption="Heatmap Overlay", use_column_width=True)

# Run AI Analysis
if st.button("🧠 Analyze X-ray with Gemini"):
    with st.spinner("Analyzing using Gemini..."):
        try:
            result_text = get_gemini_analysis(img, api_key)
            st.success("✅ Analysis complete!")
            st.subheader("🩺 AI Radiology Report")
            st.write(result_text)
            keywords = extract_keywords(result_text)
            if keywords:
                st.subheader("📌 Detected Keywords")
                st.write(", ".join(keywords))
            save_analysis(uploaded_file.name, result_text, keywords)
        except Exception as e:
            st.error(f"❌ Error: {e}")

st.markdown("---")
st.caption("⚠️ This tool is for **educational purposes only**. Always confirm with a licensed radiologist.")
