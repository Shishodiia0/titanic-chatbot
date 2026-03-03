import streamlit as st
import requests
import base64
from PIL import Image
import io

BACKEND_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(page_title="Titanic Chat Agent", layout="centered")

st.title("🚢 Titanic Dataset Chat Agent")

question = st.text_input("Ask a question about the Titanic dataset")

if st.button("Ask"):
    if question:
        response = requests.post(
            BACKEND_URL,
            json={"question": question}
        )

        data = response.json()

        st.subheader("Answer")
        st.write(data["answer"])

        if data["plot_image"]:
            image_bytes = base64.b64decode(data["plot_image"])
            image = Image.open(io.BytesIO(image_bytes))
            st.subheader("Visualization")
            st.image(image)
    else:
        st.warning("Please enter a question.")