# Import necessary libraries
import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK data (only first time)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Load spaCy model for POS tagging and NER
nlp = spacy.load("en_core_web_sm")

# -----------------------------
# 1. Define Text Cleaning Function
# -----------------------------
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    text = re.sub(r'http\S+|www\S+', '', text)  # remove URLs
    text = re.sub(r'\S+@\S+', '', text)  # remove emails
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove punctuation and numbers
    return text.strip()

# -----------------------------
# 2. Define Main Preprocessing Function
# -----------------------------
def preprocess_text(user_input):
    # Clean the text
    cleaned_text = clean_text(user_input)

    # Tokenization
    tokens = word_tokenize(cleaned_text)

    # Stopword Removal
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    # POS Tagging
    pos_tags = nltk.pos_tag(filtered_tokens)

    # Named Entity Recognition (NER)
    doc = nlp(user_input)
    named_entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Return all results
    return {
        "cleaned_text": cleaned_text,
        "tokens": filtered_tokens,
        "stems": stemmed_tokens,
        "lemmas": lemmatized_tokens,
        "pos_tags": pos_tags,
        "named_entities": named_entities
    }

# -----------------------------
# 3. Take User Input and Run Preprocessing
# -----------------------------
if __name__ == "__main__":
    user_input = input("Enter your text: ")

    result = preprocess_text(user_input)

    # Display results
    print("\n--- DATA PREPROCESSING RESULTS ---")
    print("Cleaned Text:\n", result["cleaned_text"], "\n")
    print("Tokens:\n", result["tokens"], "\n")
    print("Stems:\n", result["stems"], "\n")
    print("Lemmas:\n", result["lemmas"], "\n")
    print("Part of Speech Tags:\n", result["pos_tags"], "\n")
    print("Named Entities:")
    for ent, label in result["named_entities"]:
        print(f"  {ent} -> {label}")
