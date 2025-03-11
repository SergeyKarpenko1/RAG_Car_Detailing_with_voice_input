# Структура проекта RAG-чатбота

Этот каталог содержит компоненты для работы с RAG-чатботом, специализирующимся на автомобильной тематике.

## Структура каталогов

### 1. ChatBot
Файлы, связанные с работой чат-бота и моделями:
- `chat_bot.ipynb` - интерактивный чат-бот в Jupyter Notebook
- `openrouter_loader.py` - загрузка моделей OpenRouter
- `check_openrouter_models.py` - проверка доступных моделей OpenRouter
- `test_speech_recognition.py` - тестирование распознавания речи

### 2. ChromaDB
Файлы для работы с векторной базой данных ChromaDB:
- `check_chroma_db.py` - проверка содержимого базы данных ChromaDB
- `check_chroma_collections.py` - проверка коллекций в базе данных ChromaDB
- `check_chroma_collections2.py` - альтернативная версия проверки коллекций
- `load_data_to_chroma.py` - загрузка данных в ChromaDB
- `load_data_to_chroma_once.py` - однократная загрузка данных в ChromaDB

### 3. Utils
Вспомогательные утилиты:
- `clean_links.ipynb` - очистка ссылок
- `search_tool.py` - инструмент для поиска

### 4. chatbot_modules
Основные модули чат-бота:
- `app.py` - Streamlit-приложение для чат-бота
- `chat.py` - основной класс чат-бота
- `db.py` - работа с базой данных
- `retriever.py` - компонент для поиска релевантных документов
- `speech_to_text.py` - распознавание речи
- и другие вспомогательные модули

### 5. Scraping
Скрипты для скрапинга данных:
- `main.py` - основной скрипт для скрапинга
- `main_one_page.py` - скрапинг одной страницы
- `Clean_data.ipynb` - очистка собранных данных
- `config.py` - конфигурация для скрапинга

### 6. chroma_langchain_db
База данных ChromaDB для хранения векторных представлений документов.

## Использование

Основное приложение чат-бота можно запустить с помощью:

```bash
cd chatbot_modules
streamlit run app.py
```

Для загрузки данных в векторную базу используйте скрипты из папки ChromaDB.