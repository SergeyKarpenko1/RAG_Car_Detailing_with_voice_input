"""
Модульный чат-бот с использованием Retrieval-Augmented Generation (RAG).

Этот пакет содержит модули для создания чат-бота, который использует
базу знаний и языковую модель для генерации ответов на вопросы пользователя.

Модули:
- retriever: Модуль для поиска документов в базе знаний
- llm: Модуль для работы с языковыми моделями
- chat: Модуль для управления диалогом
- utils: Вспомогательные функции
- main: Основной файл для запуска чат-бота
"""

from .retriever import DocumentRetriever
from .chat import RAGChatBot
from .utils import ChatHistory, ResponseCache, search_web

__all__ = [
    'DocumentRetriever',
    'RAGChatBot',
    'ChatHistory',
    'ResponseCache',
    'search_web'
]