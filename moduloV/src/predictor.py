import os
import joblib
import pickle
import datetime
import numpy as np
from utils import clean_text, lemmatize_pipe

class Predictor:
    def __init__(self, model_dir="../models/"):
        # Cargar vectorizador y modelos
        self.vectorizer = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))
        self.clf = pickle.load(open(os.path.join(model_dir, "regression_model.pkl"), "rb"))
        self.svd = joblib.load(os.path.join(model_dir, "svd_model.pkl"))
        self.km = joblib.load(os.path.join(model_dir, "kmeans_model.pkl"))

        # Diccionarios
        self.sentiment_mapping = {0:"NEUTRAL", 1:"POSITIVE", 2:"NEGATIVE", 3:"MIXED"}
        self.cluster_names = {0:"Aficionados", 1:"Fifas", 2:"Fanáticos"}

    def predict(self, tweet_text):
        # Limpieza + lematización
        clean = clean_text(tweet_text)
        clean_lemmas = lemmatize_pipe([clean])[0]

        # Vectorización
        X_input = self.vectorizer.transform([clean_lemmas])

        # Reducción con SVD
        X_reduced = self.svd.transform(X_input)

        # --- Supervisado ---
        proba = self.clf.predict_proba(X_input)[0]
        pred_class = self.clf.predict(X_input)[0]

        # --- No supervisado ---
        cluster_label = self.km.predict(X_reduced)[0]
        cluster_name = self.cluster_names.get(cluster_label, str(cluster_label))

        # --- Construcción salida ---
        output = {
            "status": "success",
            "data": {
                "timestamp": datetime.datetime.now().strftime("%d/%m/%YT%H:%M:%S"),
                "student_name": "Erika Margarita Villalobos Martinez",
                "proba_positive": float(proba[1]),
                "proba_negative": float(proba[2]),
                "proba_neutral": float(proba[0]),
                "proba_mixed": float(proba[3]),
                "class": self.sentiment_mapping[pred_class],
                "cluster": cluster_name
            }
        }
        return output
