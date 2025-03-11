# Инструкция по настройке Docker-контейнера для чат-бота на Streamlit

Данная инструкция описывает процесс настройки и запуска чат-бота на базе Streamlit в Docker-контейнере.

## Содержание

1. [Требования](#требования)
2. [Структура проекта](#структура-проекта)
3. [Настройка Dockerfile](#настройка-dockerfile)
4. [Настройка docker-compose.yml](#настройка-docker-composeyml)
5. [Запуск контейнера](#запуск-контейнера)
6. [Проверка работоспособности](#проверка-работоспособности)
7. [Решение проблем](#решение-проблем)

## Требования

- Docker
- Docker Compose
- Python 3.10 или выше
- API ключи для внешних сервисов (OpenRouter, Serper)

## Структура проекта

```
project/
├── .env                      # Файл с переменными окружения
├── Dockerfile                # Инструкции для сборки Docker-образа
├── docker-compose.yml        # Конфигурация Docker Compose
├── app_docker.py             # Основной файл приложения для Docker
├── chroma_langchain_db/      # Директория с базой данных ChromaDB
│   └── knowledge/            # Поддиректория с базой знаний
├── chat_history.db           # База данных SQLite для хранения истории чата
└── response_cache.json       # Файл для кэширования ответов
```

## Настройка Dockerfile

Создайте файл `Dockerfile` со следующим содержимым:

```dockerfile
FROM python:3.10-slim

# Установка рабочей директории
WORKDIR /app

# Установка зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    python3-pyaudio \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Копирование файла с зависимостями
COPY Notebooks/chatbot_modules/requirements.txt .

# Установка зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Установка дополнительных пакетов
RUN pip install --no-cache-dir rank_bm25

# Копирование модулей чат-бота
COPY Notebooks/chatbot_modules/ /app/

# Копирование файла с переменными окружения
COPY .env /app/.env

# Копирование базы данных ChromaDB
COPY chroma_langchain_db/ /app/chroma_langchain_db/

# Создание директории для данных
RUN mkdir -p /app/data

# Копирование модифицированной версии app.py для Docker
COPY app_docker.py /app/app.py

# Создание директории для базы знаний
RUN mkdir -p /app/chroma_langchain_db/knowledge

# Запуск приложения
CMD ["streamlit", "run", "app.py"]
```

## Настройка docker-compose.yml

Создайте файл `docker-compose.yml` со следующим содержимым:

```yaml
services:
  chatbot:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: streamlit-chatbot
    ports:
      - "8502:8501"
    volumes:
      # Монтируем локальную директорию с базой данных ChromaDB
      - ./chroma_langchain_db:/app/chroma_langchain_db
      # Убедимся, что директория knowledge существует
      - ./chroma_langchain_db/knowledge:/app/chroma_langchain_db/knowledge
      # Монтируем файлы для сохранения истории чата и кэша
      - ./chat_history.db:/app/chat_history.db
      - ./response_cache.json:/app/response_cache.json
      # Монтируем .env файл с API ключами
      - ./.env:/app/.env
    environment:
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
      - SERPER_API_KEY=${SERPER_API_KEY}
    restart: unless-stopped
```

## Запуск контейнера

1. Убедитесь, что у вас есть файл `.env` с необходимыми API ключами:

```
OPENROUTER_API_KEY=ваш_ключ_openrouter
SERPER_API_KEY=ваш_ключ_serper
```

2. Соберите и запустите контейнер:

```bash
docker compose build
docker compose up -d
```

3. Проверьте, что контейнер запущен:

```bash
docker ps
```

## Проверка работоспособности

1. Откройте браузер и перейдите по адресу: http://localhost:8502
2. Введите вопрос в поле ввода и нажмите "Отправить"
3. Проверьте, что чат-бот отвечает на вопросы

## Решение проблем

### Проблема: Не работает поиск по базе данных

Если чат-бот не находит информацию в базе данных, проверьте:
1. Правильно ли смонтирована директория с базой данных: `docker inspect streamlit-chatbot | grep -A 10 "Mounts"`
2. Существует ли директория `chroma_langchain_db/knowledge` в корне проекта
3. Содержит ли директория `chroma_langchain_db/knowledge` файлы базы данных ChromaDB
4. Установлен ли пакет rank_bm25: если вы видите ошибку "не удалось импортировать rank_bm25", добавьте в Dockerfile:
   ```
   RUN pip install --no-cache-dir rank_bm25
   ```
   и пересоберите образ: `docker compose build && docker compose up -d`

### Проблема: Не работает поиск в интернете

Если чат-бот не выполняет поиск в интернете, проверьте:
1. Указан ли API ключ SERPER_API_KEY в файле .env
2. Передается ли переменная окружения SERPER_API_KEY в контейнер (проверьте раздел environment в docker-compose.yml)
3. Доступен ли сервис Serper API (проверьте логи контейнера: `docker logs streamlit-chatbot`)

### Проблема: Чат-бот отвечает на разные вопросы одинаково

Если чат-бот отвечает на разные вопросы одинаково, проблема может быть в кэшировании ответов. Решения:

1. Очистите кэш ответов:
   ```bash
   echo "{}" > response_cache.json
   ```

2. Отключите использование кэша в app_docker.py, изменив вызов метода answer:
   ```python
   response = st.session_state.chatbot.answer(user_input, user_id=st.session_state.user_id, use_cache=False)
   ```

3. Модифицируйте класс ResponseCache в файле utils.py для более точного сравнения вопросов:
   ```python
   def get_response(self, question: str) -> Optional[Dict[str, Any]]:
       """
       Получение ответа из кэша с более точным сравнением вопросов.
       """
       # Нормализация вопроса
       normalized_question = question.lower().strip()
       
       # Точное совпадение
       if normalized_question in self.cache:
           return self.cache[normalized_question]
       
       return None
   ```

### Проблема: Контейнер не запускается

Если контейнер не запускается, проверьте:
1. Логи контейнера: `docker logs streamlit-chatbot`
2. Доступность портов: убедитесь, что порт 8502 не занят другим приложением
3. Права доступа к файлам: убедитесь, что у пользователя docker есть права на чтение и запись файлов, монтируемых в контейнер

## Дополнительные команды

### Перезапуск контейнера

```bash
docker compose restart
```

### Остановка контейнера

```bash
docker compose down
```

### Просмотр логов

```bash
docker logs streamlit-chatbot
```

### Вход в контейнер

```bash
docker exec -it streamlit-chatbot bash