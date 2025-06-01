import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from model import load_model_and_vectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

def evaluate_model():
    # Charger modèle et vectorizer
    model, vectorizer, threshold = load_model_and_vectorizer()

    # Charger les données pour test
    df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'text']
    df = df.dropna()
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    X = df['text']
    y = df['label']
    _, X_test_raw, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    X_test_vect = vectorizer.transform(X_test_raw)
    y_prob = model.predict_proba(X_test_vect)[:, 1]
    y_pred = (y_prob > threshold).astype(int)

    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ham', 'Spam'])
    disp.plot(cmap='Blues')
    plt.title("Matrice de Confusion - Détection de SPAM")
    plt.show()

if __name__ == "__main__":
    evaluate_model()
