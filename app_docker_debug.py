#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Отладочная версия Streamlit-приложения для чат-бота.
"""

import os
import streamlit as st
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Настройка страницы Streamlit
st.set_page_config(
    page_title="Debug Chatbot",
    page_icon="🚗",
    layout="wide"
)

# Отображение отладочной информации
st.title("Отладочная информация")

# Проверка переменных окружения
st.header("Переменные окружения")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

st.write(f"OPENROUTER_API_KEY: {'✅ Найден' if openrouter_api_key else '❌ Не найден'}")
st.write(f"SERPER_API_KEY: {'✅ Найден' if serper_api_key else '❌ Не найден'}")

# Проверка директорий
st.header("Директории")
chroma_dir = "/app/chroma_langchain_db"
knowledge_dir = "/app/chroma_langchain_db/knowledge"

st.write(f"Директория ChromaDB: {'✅ Существует' if os.path.exists(chroma_dir) else '❌ Не существует'}")
st.write(f"Директория Knowledge: {'✅ Существует' if os.path.exists(knowledge_dir) else '❌ Не существует'}")

# Проверка файлов
st.header("Файлы")
env_file = "/app/.env"
chat_history_db = "/app/chat_history.db"
response_cache = "/app/response_cache.json"

st.write(f"Файл .env: {'✅ Существует' if os.path.exists(env_file) else '❌ Не существует'}")
st.write(f"Файл chat_history.db: {'✅ Существует' if os.path.exists(chat_history_db) else '❌ Не существует'}")
st.write(f"Файл response_cache.json: {'✅ Существует' if os.path.exists(response_cache) else '❌ Не существует'}")

# Проверка содержимого директории knowledge
st.header("Содержимое директории knowledge")
if os.path.exists(knowledge_dir):
    files = os.listdir(knowledge_dir)
    st.write(f"Количество файлов: {len(files)}")
    for file in files:
        st.write(f"- {file}")
else:
    st.write("Директория knowledge не существует")

# Проверка файлов chat_history.db и response_cache.json
st.header("Проверка файлов")
st.write(f"Файл chat_history.db: {'✅ Существует' if os.path.exists(chat_history_db) else '❌ Не существует'}")
st.write(f"Размер файла chat_history.db: {os.path.getsize(chat_history_db) if os.path.exists(chat_history_db) else 'Файл не существует'} байт")
st.write(f"Файл response_cache.json: {'✅ Существует' if os.path.exists(response_cache) else '❌ Не существует'}")
st.write(f"Размер файла response_cache.json: {os.path.getsize(response_cache) if os.path.exists(response_cache) else 'Файл не существует'} байт")

# Содержимое директории /app
st.header("Содержимое директории /app")
app_files = os.listdir("/app")
st.write(f"Количество файлов: {len(app_files)}")
for file in app_files:
    st.write(f"- {file}")

# Форма для тестирования
st.header("Тестирование формы")
with st.form(key="test_form"):
    user_input = st.text_input("Введите текст:")
    submit_button = st.form_submit_button("Отправить")

if submit_button and user_input:
    st.success(f"Форма отправлена успешно! Введенный текст: {user_input}")
    
# Тестирование сессии
st.header("Тестирование сессии")
if "counter" not in st.session_state:
    st.session_state.counter = 0
    
if st.button("Увеличить счетчик"):
    st.session_state.counter += 1
    
st.write(f"Значение счетчика: {st.session_state.counter}")