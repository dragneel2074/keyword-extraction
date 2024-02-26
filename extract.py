
import streamlit as st
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download NLTK data (if not already present)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Load the models and feature names


# Preprocessing function similar to what was used in notebook
def preprocess_text(text):
    text = text.lower()
    #text = re.sub(r'<.*?>', '', text)
    #text = re.sub(r'http\S+', '', text)
    #text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text)
    text = " ".join([w for w in words if not w in stop_words])
    #lem = WordNetLemmatizer()
    #text = " ".join([lem.lemmatize(w) for w in text.split()])
    return text

# Streamlit app
st.title('Keyword Extraction App')

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    text = str(uploaded_file.read(), 'utf-8')  # Assuming the file is a text file
    preprocessed_text = preprocess_text(text)
    cv = pickle.load(open('cv.pkl', 'rb'))
    tf = pickle.load(open('tf.pkl', 'rb'))
    feature_names = pickle.load(open('feature_names.pkl', 'rb'))

    # Vectorize the text
    cv_transformed = cv.transform([preprocessed_text])
    tfidf_transformed = tf.transform(cv_transformed)

    # Extract and display keywords
    sorted_items = zip(tfidf_transformed.tocoo().col, tfidf_transformed.tocoo().data)
    sorted_items = sorted(sorted_items, key=lambda x: (x[1], x[0]), reverse=True)

    top_n = 10
    top_keywords = [(feature_names[idx], score) for idx, score in sorted_items[:top_n]]

    st.write("Top keywords in your document:")
    for keyword, score in top_keywords:
        st.write(f"{keyword}: {score}")
