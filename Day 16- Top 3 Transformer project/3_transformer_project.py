import streamlit as st
from transformers import pipeline

# ---------------------- Custom CSS Styling ----------------------
st.markdown("""
    <style>
        /* Page background and text */
        .stApp {
            background-color: #f5f7ff;
            color: #2c2c54;
            font-family: 'Poppins', sans-serif;
        }

        /* Title style */
        h1 {
            color: #4b7bec;
            text-align: center;
            font-weight: 700;
            margin-bottom: 0.5em;
        }

        /* Subheaders */
        h2, h3 {
            color: #706fd3;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #4b7bec;
            color: white;
        }
        [data-testid="stSidebar"] * {
            color: white !important;
        }

        /* Text area and widgets */
        textarea {
            border-radius: 12px !important;
            border: 2px solid #706fd3 !important;
        }

        /* Button styling */
        div.stButton > button {
            background-color: #706fd3;
            color: white;
            border-radius: 10px;
            border: none;
            font-size: 16px;
            padding: 0.5em 1em;
            transition: 0.3s;
        }
        div.stButton > button:hover {
            background-color: #4b7bec;
            transform: scale(1.05);
        }

        /* Result box */
        .result-box {
            background-color: white;
            padding: 1em;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(112, 111, 211, 0.2);
            margin-top: 1em;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- NLP Task Functions ----------------------
def summarize_text(input_text):
    summarizer = pipeline("summarization")
    summary = summarizer(input_text, max_length=150, min_length=50, length_penalty=2.0)[0]['summary_text']
    return summary

def classify_text(input_text):
    classifier = pipeline("sentiment-analysis")
    classification = classifier(input_text)[0]['label']
    return classification

def rephrase_text(input_text):
    rephraser = pipeline("text2text-generation", model="facebook/bart-large-cnn")
    rephrased_text = rephraser(input_text, max_length=100, min_length=20)[0]['generated_text']
    return rephrased_text

# ---------------------- Streamlit App Layout ----------------------
st.title("💫 Hugging Face Transformers App")

# Sidebar
selected_task = st.sidebar.selectbox(
    "Select NLP Task:",
    ["Summarization", "Text Classification", "Text Rephrasing"]
)

# Input text
input_text = st.text_area("📝 Enter your text below:")

# Button trigger
if st.button("✨ Process"):
    if selected_task == "Summarization" and input_text:
        st.subheader("📘 Summary:")
        result = summarize_text(input_text)
        st.markdown(f"<div class='result-box'>{result}</div>", unsafe_allow_html=True)

    elif selected_task == "Text Classification" and input_text:
        st.subheader("💬 Classification Result:")
        result = classify_text(input_text)
        st.markdown(f"<div class='result-box'>The text is classified as: <b>{result}</b></div>", unsafe_allow_html=True)

    elif selected_task == "Text Rephrasing" and input_text:
        st.subheader("🔁 Rephrased Text:")
        result = rephrase_text(input_text)
        st.markdown(f"<div class='result-box'>{result}</div>", unsafe_allow_html=True)

    else:
        st.info("Please enter text and select a task from the sidebar.")
