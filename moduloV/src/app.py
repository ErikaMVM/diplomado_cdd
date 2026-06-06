import streamlit as st
import datetime
from predictor import Predictor

# Inicializar predictor
predictor = Predictor(model_dir="models")

st.title("🔮 Predictor de Tweets")

tweet_text = st.text_area("Escribe un tweet para analizar:")

if st.button("Predecir"):
    with st.status("Procesando tweet...", expanded=True) as status:
        st.write("🧹 Limpieza de texto...")
        clean = predictor.vectorizer.build_tokenizer()(tweet_text)  # solo para mostrar

        st.write("🔠 Vectorización TF-IDF...")
        # Aquí no mostramos el vector completo, solo el paso

        st.write("📉 Reducción de dimensionalidad (SVD)...")
        # Paso de reducción

        st.write("🤖 Inferencia supervisada...")
        # Paso de clasificación

        st.write("📊 Asignación de cluster...")
        # Paso de clustering

        # Ejecutar realmente la predicción
        result = predictor.predict(tweet_text)

        status.update(label="✅ Proceso completado", state="complete", expanded=False)

    st.subheader("Resultado")
    st.json(result)