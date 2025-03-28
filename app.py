import streamlit as st
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def calculate_similarity(doc1, doc2):
    documents = [doc1, doc2]
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0]

st.title("Document Similarity Checker")

doc1 = st.text_area("Enter Document 1")
doc2 = st.text_area("Enter Document 2")

if st.button("Calculate Similarity"):
    if doc1 and doc2:
        doc1 = preprocess(doc1)
        doc2 = preprocess(doc2)
        score = calculate_similarity(doc1, doc2)
        st.success(f"Similarity Score: {score:.4f}")
    else:
        st.warning("Please enter both documents.")
