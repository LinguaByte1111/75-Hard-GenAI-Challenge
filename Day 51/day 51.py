import streamlit as st
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import google.generativeai as genai

# Configure API Key
GOOGLE_API_KEY = "YOUR_API_KEY_HERE"
genai.configure(api_key=GOOGLE_API_KEY)
vision_model = genai.GenerativeModel("gemini-pro-vision")


@st.cache_data
def fetch_image(image_url):
    try:
        result = requests.get(image_url)
        image = Image.open(BytesIO(result.content))
        return image
    except UnidentifiedImageError:
        st.error("Error: Could not recognize this image format.")
        return None


def main():
    st.title("AI-Powered Research Figure Reviewer")

    st.sidebar.header("Image Input")
    image_url = st.sidebar.text_input("Enter Image URL")

    mode = st.sidebar.selectbox(
        "Choose Mode",
        [
            "General Chat About Image",
            "Zero-Shot Figure Review"
        ]
    )

    image = None
    if image_url:
        image = fetch_image(image_url)
        if image:
            st.sidebar.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("Provide your question or request below")
    user_input = st.text_input("Enter your query")

    if st.button("Submit"):
        if not image:
            st.error("Please provide a valid image URL first.")
            return

        if mode == "Zero-Shot Figure Review":
            final_prompt = (
                "You are an expert scientific reviewer. "
                "Analyze this research figure as if reviewing a paper. "
                "Evaluate statistical correctness, clarity, fairness of baselines, "
                "methodological soundness, axis integrity, readability, and reproducibility. "
                "Suggest improvements.\n\n"
                f"User request: {user_input}"
            )
        else:
            final_prompt = user_input

        response = vision_model.generate_content([final_prompt, image])

        st.subheader("Model Response")
        st.text_area("Output", value=response.text, height=300)


if __name__ == "__main__":
    main()
