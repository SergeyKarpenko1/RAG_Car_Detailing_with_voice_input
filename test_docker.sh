#!/bin/bash

# Скрипт для тестирования Docker-контейнера с чат-ботом

echo "Проверка статуса контейнера..."
CONTAINER_STATUS=$(docker ps -f name=streamlit-chatbot --format "{{.Status}}")

if [ -z "$CONTAINER_STATUS" ]; then
    echo "Контейнер не запущен. Запускаем..."
    docker compose up -d
    sleep 5
else
    echo "Контейнер запущен: $CONTAINER_STATUS"
fi

echo "Проверка доступности API ключей..."
docker exec streamlit-chatbot bash -c 'echo "OPENROUTER_API_KEY: $(if [ -n \"$OPENROUTER_API_KEY\" ]; then echo \"найден\"; else echo \"не найден\"; fi)"'
docker exec streamlit-chatbot bash -c 'echo "SERPER_API_KEY: $(if [ -n \"$SERPER_API_KEY\" ]; then echo \"найден\"; else echo \"не найден\"; fi)"'

echo "Проверка наличия файлов и директорий..."
docker exec streamlit-chatbot bash -c 'echo "Файл .env: $(if [ -f /app/.env ]; then echo \"существует\"; else echo \"не существует\"; fi)"'
docker exec streamlit-chatbot bash -c 'echo "Директория ChromaDB: $(if [ -d /app/chroma_langchain_db ]; then echo \"существует\"; else echo \"не существует\"; fi)"'
docker exec streamlit-chatbot bash -c 'echo "Директория Knowledge: $(if [ -d /app/chroma_langchain_db/knowledge ]; then echo \"существует\"; else echo \"не существует\"; fi)"'
docker exec streamlit-chatbot bash -c 'echo "Файл chat_history.db: $(if [ -f /app/chat_history.db ]; then echo \"существует\"; else echo \"не существует\"; fi)"'
docker exec streamlit-chatbot bash -c 'echo "Файл response_cache.json: $(if [ -f /app/response_cache.json ]; then echo \"существует\"; else echo \"не существует\"; fi)"'

echo "Проверка размера кэша ответов..."
docker exec streamlit-chatbot bash -c 'echo "Размер response_cache.json: $(wc -c < /app/response_cache.json) байт"'

echo "Чат-бот доступен по адресу: http://localhost:8502"