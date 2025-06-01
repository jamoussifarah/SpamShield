import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

def load_model_and_vectorizer():
    # Charger et préparer les données
    df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'text']
    df = df.dropna()
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    X = df['text']
    y = df['label']
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=5)
    X_train_vect = vectorizer.fit_transform(X_train_raw)

    model = MultinomialNB()
    model.fit(X_train_vect, y_train)

    # Calcul du seuil optimal basé sur F1-score
    y_prob = model.predict_proba(vectorizer.transform(X_test_raw))[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.01)
    from sklearn.metrics import f1_score
    f1_scores = [f1_score(y_test, (y_prob > t).astype(int)) for t in thresholds]
    best_thresh = thresholds[np.argmax(f1_scores)]

    return model, vectorizer, best_thresh

def predict_spam(model, vectorizer, threshold, message):
    vect = vectorizer.transform([message])
    prob = model.predict_proba(vect)[:, 1][0]
    prediction = 1 if prob > threshold else 0
    return prediction, prob
