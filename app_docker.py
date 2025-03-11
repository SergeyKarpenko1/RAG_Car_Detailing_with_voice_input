#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit-приложение для чат-бота с RAG и голосовым вводом.
Модифицированная версия для запуска в Docker-контейнере.
"""

# Отключаем параллелизм в tokenizers для предотвращения дедлоков
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Импортируем необходимые библиотеки
import uuid
import time
import sys
import tempfile
import queue
import threading
import av
from dotenv import load_dotenv

# Добавляем текущую директорию в путь поиска модулей
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем streamlit после установки переменных окружения
import streamlit as st

# Импортируем модули из текущей директории
from db import ChatDatabase
from chat import RAGChatBot
from retriever import DocumentRetriever
from speech_to_text import SpeechRecognizer

# Загрузка переменных окружения
load_dotenv()

# Настройка страницы Streamlit
st.set_page_config(
    page_title="Detailing your car",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Функция для инициализации сессии
def init_session():
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "speech_recognizer" not in st.session_state:
        try:
            st.session_state.speech_recognizer = SpeechRecognizer(
                recognizer_type="google",
                language="ru-RU"
            )
        except Exception as e:
            st.error(f"Ошибка при инициализации распознавателя речи: {e}")
            st.session_state.speech_recognizer = None
    
    if "chatbot" not in st.session_state:
        try:
            # Проверка наличия API ключа
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                st.error("API ключ не найден. Укажите его в .env файле или переменной окружения.")
                st.stop()
            
            # Создание модели
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                openai_api_key=api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                model_name="anthropic/claude-3.5-haiku",
                temperature=0.5,
                max_tokens=2048,
            )
            
            # Путь к файлу базы данных SQLite для хранения истории диалогов
            db_path = "./chat_history.db"
            
            # Создание ретривера с реранкером
            # Оборачиваем в try-except для обработки возможных ошибок
            try:
                # В Docker-контейнере используем фиксированный путь к базе данных
                persist_directory = "/app/chroma_langchain_db/knowledge"
                
                print(f"Путь к базе данных в Docker: {persist_directory}")
                
                retriever = DocumentRetriever(
                    persist_directory=persist_directory,
                    collection_name="knowledge_markdown",
                    embedding_model_name="intfloat/multilingual-e5-base",
                    similarity_top_k=10,
                    bm25_top_k=10,
                    similarity_weight=0.5,
                    bm25_weight=0.5,
                    use_reranker=True,
                    reranker_model_name="DiTy/cross-encoder-russian-msmarco",
                    reranker_top_k=5
                )
            except Exception as e:
                st.error(f"Ошибка при инициализации ретривера: {e}")
                # Создаем упрощенный ретривер без реранкера
                persist_directory = "/app/chroma_langchain_db/knowledge"
                
                print(f"Путь к базе данных в Docker (fallback): {persist_directory}")
                
                retriever = DocumentRetriever(
                    persist_directory=persist_directory,
                    collection_name="knowledge_markdown",
                    embedding_model_name="intfloat/multilingual-e5-base",
                    use_reranker=False
                )
            
            # Создание чат-бота
            st.session_state.chatbot = RAGChatBot(
                llm=llm,
                retriever=retriever,
                cache_file="./response_cache.json",
                db_path=db_path,
                use_web_search=True,
                max_web_search_attempts=3,
                max_relevant_sources=3,
                use_voice_input=True,
                speech_recognizer_type="google",
                speech_language="ru-RU"
            )
        except Exception as e:
            st.error(f"Ошибка при инициализации чат-бота: {e}")
            import traceback
            st.error(f"Детали ошибки: {traceback.format_exc()}")

# Функция для загрузки истории диалога из базы данных
def load_chat_history():
    if "db" not in st.session_state:
        st.session_state.db = ChatDatabase("./chat_history.db")
    
    history = st.session_state.db.get_chat_history(st.session_state.user_id)
    
    # Обновляем сообщения в сессии
    st.session_state.messages = []
    # Загружаем историю в обратном порядке, чтобы последние сообщения были в начале списка
    for msg in reversed(history):
        role = "user" if msg["role"] == "user" else "assistant"
        st.session_state.messages.append({"role": role, "content": msg["message"]})

# Функция для очистки истории диалога
def clear_chat_history():
    if "db" not in st.session_state:
        st.session_state.db = ChatDatabase("./chat_history.db")
    
    st.session_state.db.clear_history(st.session_state.user_id)
    st.session_state.messages = []
    st.rerun()

# Определение опций распознавателя речи
recognizer_options = {
    "google": "Google Speech Recognition",
    "whisper": "OpenAI Whisper (локальный)",
    "whisper_api": "OpenAI Whisper API",
    "sphinx": "CMU Sphinx (офлайн)"
}

# Инициализация переменных состояния для выбора распознавателя
if "recognizer_type" not in st.session_state:
    st.session_state.recognizer_type = "google"

# Инициализация сессии
init_session()

# Боковая панель
with st.sidebar:
    st.title("🚗 Автомобильный чат-бот")
    st.markdown("""
    ### О боте
    
    Этот чат-бот специализируется на вопросах об уходе за автомобилем.
    
    **Особенности:**
    - Комбинированный поиск по базе знаний (векторный + BM25)
    - Переранжирование результатов с помощью CrossEncoder
    - Хранение истории диалогов в SQLite базе данных
    - Поиск в интернете, если информации в базе знаний недостаточно
    - Голосовой ввод с использованием Google Speech Recognition
    
    **Команды:**
    - Нажмите на кнопку микрофона для голосового ввода
    - Используйте кнопку "Очистить историю" для удаления истории диалога
    """)
    
    st.divider()
    
    # Кнопка для очистки истории диалога
    if st.button("Очистить историю", key="clear_history"):
        clear_chat_history()
    
    # Информация о пользователе
    st.info(f"ID пользователя: {st.session_state.user_id}")

# Основная часть
st.title("Автомобильный чат-бот")

# Отображение сообщений перенесено в messages_container
# Этот блок больше не нужен, так как мы отображаем сообщения в контейнере выше

# Форма для ввода вопроса - всегда отображается вверху
input_container = st.container()

# Индикатор "думаю" - будет отображаться во время обработки запроса
thinking_container = st.container()

# Отображение сообщений
messages_container = st.container()

# Отображаем сообщения в контейнере для сообщений
# Сообщения уже добавляются в начало списка, поэтому порядок правильный
with messages_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Форма для ввода вопроса в контейнере вверху
with input_container:
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    
    # Используем форму для размещения элементов
    with st.form(key="input_form", clear_on_submit=False):
        # Поле ввода на всю ширину
        user_input = st.text_input(
            "Задайте вопрос:",
            value=st.session_state.user_input,
            key="input_field",
            placeholder="Например: Как правильно мыть двигатель автомобиля?"
        )
        
        # Кнопки под полем ввода, выровненные по левому краю
        col1, col2, col3 = st.columns([2, 2, 6])
        
        with col1:
            # Кнопка отправки запроса
            submit_button = st.form_submit_button("Отправить", type="primary")
            
        with col2:
            # Кнопка голосового ввода
            speech_button = st.form_submit_button("🎤", help="Голосовой ввод")
            
        with col3:
            # Пустая колонка для заполнения оставшегося пространства
            pass
        
    # Обработка отправки формы
    if submit_button and user_input:
        # Добавляем сообщение пользователя в начало списка
        st.session_state.messages.insert(0, {"role": "user", "content": user_input})
        
        # Отображаем индикатор "думаю"
        with thinking_container:
            st.info("🤔 Думаю... Ищу информацию и формирую ответ...")
        
        # Получаем ответ от чат-бота напрямую
        with st.spinner("Обработка запроса..."):
            start_time = time.time()
            response = st.session_state.chatbot.answer(user_input, user_id=st.session_state.user_id)
            elapsed_time = time.time() - start_time
        
        # Формируем полный ответ бота с источниками и метаданными
        bot_answer = response["answer"]
        
        # Добавляем источники, если они есть
        if response["sources"]:
            bot_answer += "\n\n**Источники:**"
            for source in response["sources"]:
                bot_answer += f"\n- {source}"
        
        # Добавляем информацию о времени выполнения и использовании веб-поиска
        bot_answer += f"\n\n*Время выполнения: {elapsed_time:.2f} сек.*"
        if response["used_web_search"]:
            bot_answer += "\n*Использован поиск в интернете*"
        if response["from_cache"]:
            bot_answer += "\n*Ответ получен из кэша*"
        
        # Добавляем ответ бота в начало списка (после сообщения пользователя)
        st.session_state.messages.insert(1, {"role": "assistant", "content": bot_answer})
        
        # Очищаем поле ввода
        st.session_state.user_input = ""
        
        # Перезагружаем страницу для отображения обновленных сообщений
        st.rerun()

# Если нажата кнопка голосового ввода, показываем интерфейс записи
if "show_voice_input" not in st.session_state:
    st.session_state.show_voice_input = False

if speech_button:
    st.session_state.show_voice_input = True

# Показываем интерфейс голосового ввода, если он активирован
speech_text = None
if st.session_state.show_voice_input:
    with input_container:
        st.subheader("🎤 Голосовой ввод")
        
        # Выбор распознавателя
        selected_recognizer = st.selectbox(
            "Выберите распознаватель:",
            options=list(recognizer_options.keys()),
            format_func=lambda x: recognizer_options[x],
            index=list(recognizer_options.keys()).index(st.session_state.recognizer_type),
            key="recognizer_select"
        )
        
        # Обновляем тип распознавателя в сессии
        if selected_recognizer != st.session_state.recognizer_type:
            st.session_state.recognizer_type = selected_recognizer
            # Пересоздаем распознаватель с новым типом
            try:
                st.session_state.speech_recognizer = SpeechRecognizer(
                    recognizer_type=selected_recognizer,
                    language="ru-RU"
                )
                st.success(f"Распознаватель изменен на {recognizer_options[selected_recognizer]}")
            except Exception as e:
                st.error(f"Ошибка при инициализации распознавателя: {e}")
        
        # Запись голосового сообщения
        audio_data = st.audio_input("Нажмите для записи голосового сообщения", key="audio_input")
        
        # Если есть аудио данные, обрабатываем их
        if audio_data:
            with st.spinner("Распознаю речь..."):
                try:
                    # Сохраняем аудио во временный файл
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        temp_filename = temp_file.name
                        temp_file.write(audio_data.getvalue())
                    
                    # Распознаем речь из файла
                    text, confidence = st.session_state.speech_recognizer.recognize_from_file(temp_filename)
                    
                    # Удаляем временный файл
                    os.unlink(temp_filename)
                    
                    if text and not text.startswith("Ошибка") and not text.startswith("Таймаут"):
                        # Не устанавливаем st.session_state.user_input здесь,
                        # чтобы избежать автоматической отправки запроса
                        st.success(f"Распознано: {text}")
                        speech_text = text
                    else:
                        st.error(f"Ошибка распознавания: {text}")
                except Exception as e:
                    # Удаляем временный файл в случае ошибки
                    if 'temp_filename' in locals() and os.path.exists(temp_filename):
                        os.unlink(temp_filename)
                    st.error(f"Ошибка при распознавании речи: {e}")
                    import traceback
                    st.error(f"Детали ошибки: {traceback.format_exc()}")
        
        # Располагаем кнопки вертикально одна под другой
        col1, col2 = st.columns(2)
        with col1:
            if speech_text and st.button("Отправить запрос", key="send_voice"):
                # Скрываем интерфейс голосового ввода
                st.session_state.show_voice_input = False
                
                # Добавляем сообщение пользователя в начало списка
                st.session_state.messages.insert(0, {"role": "user", "content": speech_text})
                
                # Отображаем индикатор "думаю"
                with thinking_container:
                    st.info("🤔 Думаю... Ищу информацию и формирую ответ...")
                
                # Получаем ответ от чат-бота напрямую
                with st.spinner("Обработка запроса..."):
                    start_time = time.time()
                    response = st.session_state.chatbot.answer(speech_text, user_id=st.session_state.user_id)
                    elapsed_time = time.time() - start_time
                
                # Формируем полный ответ бота с источниками и метаданными
                bot_answer = response["answer"]
                
                # Добавляем источники, если они есть
                if response["sources"]:
                    bot_answer += "\n\n**Источники:**"
                    for source in response["sources"]:
                        bot_answer += f"\n- {source}"
                
                # Добавляем информацию о времени выполнения и использовании веб-поиска
                bot_answer += f"\n\n*Время выполнения: {elapsed_time:.2f} сек.*"
                if response["used_web_search"]:
                    bot_answer += "\n*Использован поиск в интернете*"
                if response["from_cache"]:
                    bot_answer += "\n*Ответ получен из кэша*"
                
                # Добавляем ответ бота в начало списка (после сообщения пользователя)
                st.session_state.messages.insert(1, {"role": "assistant", "content": bot_answer})
                
                # Перезагружаем страницу для отображения обновленных сообщений
                st.rerun()
        
        with col2:
            if st.button("Скрыть голосовой ввод", key="hide_voice"):
                st.session_state.show_voice_input = False
                st.rerun()

# Загрузка истории диалога при первом запуске
if len(st.session_state.messages) == 0:
    load_chat_history()