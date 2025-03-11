#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Пример использования функциональности хранения истории диалогов в чат-боте.
"""

import os
import sys
import uuid
from dotenv import load_dotenv

# Добавляем директорию с модулями в путь поиска
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Загружаем переменные окружения из .env
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from chatbot_modules.chat import RAGChatBot
from chatbot_modules.db import ChatDatabase
from chatbot_modules.retriever import DocumentRetriever

def main():
    """Основная функция для демонстрации работы с историей диалогов."""
    
    # Проверка наличия API ключа
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("API ключ не найден. Укажите его в .env файле или переменной окружения.")
    
    # Создание ретривера с реранкером
    retriever = DocumentRetriever(
        persist_directory="./chroma_langchain_db/knowledge",
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
    
    # Создание модели
    llm = ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        model_name="anthropic/claude-3.5-haiku",  # Можно заменить на другую модель
        temperature=0.5,
        max_tokens=2048,
    )
    
    # Путь к файлу базы данных SQLite для хранения истории диалогов
    db_path = "./chat_history.db"
    
    # Создание чат-бота
    chatbot = RAGChatBot(
        llm=llm,
        retriever=retriever,
        cache_file=None,  # Не используем кэш для примера
        db_path=db_path,  # Путь к файлу базы данных SQLite для хранения истории диалогов
        use_web_search=True,  # Использовать поиск в интернете
        max_web_search_attempts=3,  # Максимальное количество попыток поиска в интернете
        max_relevant_sources=3,  # Максимальное количество релевантных источников
        use_voice_input=False,  # Не используем голосовой ввод для примера
    )
    
    # Создание уникального идентификатора пользователя
    user_id = str(uuid.uuid4())
    print(f"Создан новый пользователь с ID: {user_id}")
    
    # Пример использования чат-бота с сохранением истории диалогов
    questions = [
        "Как правильно мыть двигатель автомобиля?",
        "Какие средства лучше использовать?",
        "Как часто нужно мыть двигатель?",
    ]
    
    for question in questions:
        print(f"\nВопрос: {question}")
        response = chatbot.answer(question, user_id=user_id)
        print(f"Ответ: {response['answer']}")
    
    # Получение истории диалога из базы данных
    db = ChatDatabase(db_path)
    history = db.get_chat_history(user_id)
    
    print("\n--- История диалога ---")
    for msg in history:
        role = "Пользователь" if msg["role"] == "user" else "Бот"
        print(f"{role}: {msg['message']}")
    
    # Пример очистки истории диалога
    print("\nОчистка истории диалога...")
    db.clear_history(user_id)
    
    # Проверка, что история очищена
    history = db.get_chat_history(user_id)
    if not history:
        print("История диалога успешно очищена.")
    
    # Пример запуска интерактивного чата с сохранением истории диалогов
    print("\nЗапуск интерактивного чата с сохранением истории диалогов...")
    print("Для выхода введите 'х'")
    print("Для просмотра истории диалога введите 'и'")
    print("Для очистки истории диалога введите 'о'")
    
    # Запуск интерактивного чата с указанием идентификатора пользователя
    chatbot.run_interactive_chat(user_id=user_id)

if __name__ == "__main__":
    main()