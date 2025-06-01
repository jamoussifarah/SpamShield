import streamlit as st
from model import load_model_and_vectorizer, predict_spam

@st.cache_resource
def load_model():
    return load_model_and_vectorizer()

model, vectorizer, threshold = load_model()

st.title("Détecteur de Spam")

message = st.text_area("Entrez un message à vérifier")

if st.button("Vérifier"):
    if message.strip() == "":
        st.warning("Veuillez entrer un message.")
    else:
        pred, prob = predict_spam(model, vectorizer, threshold, message)
        label = "SPAM ❌" if pred == 1 else "HAM ✅"
        st.write(f"**Résultat :** {label}")
        st.write(f"**Probabilité spam :** {prob:.2f}")
