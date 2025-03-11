#!/bin/bash

# Скрипт для перезапуска Docker-контейнера с очисткой кэша ответов

echo "Останавливаем контейнер..."
docker compose down

echo "Очищаем кэш ответов..."
echo "{}" > response_cache.json

echo "Запускаем контейнер..."
docker compose up -d

echo "Готово! Контейнер перезапущен с очищенным кэшем."
echo "Чат-бот доступен по адресу: http://localhost:8502"