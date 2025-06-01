# 📱 SpamShield - Détection de spams SMS par Machine Learning

**SpamShield** est une application interactive développée en Python, qui permet de détecter si un message texte est un spam ou non, à l'aide d'un modèle de classification Naive Bayes.

## 🚀 Fonctionnalités

- Interface web simple et rapide (via Streamlit)
- Modèle de machine learning entraîné sur des SMS réels
- Vectorisation TF-IDF avec bigrammes pour meilleure précision
- Visualisation de la matrice de confusion
- Système de prédiction personnalisée

## 🧠 Modèle utilisé

- `Multinomial Naive Bayes` (idéal pour la classification de texte)
- `TfidfVectorizer` avec n-grammes `(1,2)` pour capturer les expressions clés
- Seuil de classification ajusté (ex: 0.3) pour améliorer la détection des spams

