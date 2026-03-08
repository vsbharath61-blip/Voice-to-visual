import streamlit as st
import requests
from PIL import Image

st.title("🎤 Voice to Visual AI")

uploaded_audio = st.file_uploader("Upload voice file (.wav or .mp3)", type=["wav", "mp3"])

if uploaded_audio:
    files = {"audio": uploaded_audio}
    
    response = requests.post(
        "http://127.0.0.1:8000/generate",
        files=files
    )
    
    result = response.json()
    
    st.write("### Recognized Text:")
    st.write(result["recognized_text"])
    
    image = Image.open(result["image_file"])
    st.image(image, caption="Generated Image")