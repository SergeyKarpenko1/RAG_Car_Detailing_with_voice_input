# Используем официальный образ Python 3.10 как базовый
FROM python:3.10-slim

# Устанавливаем рабочую директорию в контейнере
WORKDIR /app

# Устанавливаем необходимые системные зависимости для PyAudio и других библиотек
RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    python3-pyaudio \
    ffmpeg \
    flac \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Копируем файлы зависимостей
COPY Notebooks/chatbot_modules/requirements.txt .

# Устанавливаем зависимости Python
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем дополнительные зависимости, которые могут отсутствовать в requirements.txt
RUN pip install --no-cache-dir rank_bm25

# Копируем исходный код приложения
COPY Notebooks/chatbot_modules/ /app/
COPY .env /app/.env
COPY chroma_langchain_db/ /app/chroma_langchain_db/

# Создаем директории для хранения данных
RUN mkdir -p /app/data

# Устанавливаем переменные окружения
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

# Открываем порт для Streamlit
EXPOSE 8501

# Копируем модифицированную версию app.py для Docker
COPY app_docker.py /app/app.py

# Создаем директорию для базы данных ChromaDB
RUN mkdir -p /app/chroma_langchain_db/knowledge

# Устанавливаем точку входа для запуска Streamlit-приложения
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]