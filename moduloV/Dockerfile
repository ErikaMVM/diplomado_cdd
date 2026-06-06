FROM python:3.11-slim

WORKDIR /app

# Copiar dependencias
COPY requirements.txt .

# Instalar dependencias
RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt stopwords

# Copiar código fuente y modelos
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8080

# Comando de inicio: Streamlit
CMD ["streamlit", "run", "src/app.py", "--server.port=8080", "--server.address=0.0.0.0"]