#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для однократной загрузки данных в ChromaDB.
Этот скрипт загружает данные в базу данных ChromaDB и сохраняет их.
База данных будет использоваться приложением app.py.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import re
import sys

def extract_links(text: str) -> str:
    """
    Извлекает ссылки (URL) из текста и возвращает их в виде строки.
    """
    url_pattern = re.compile(r'https?://[^\s)]+')
    links = url_pattern.findall(text)
    return ", ".join(links)

def load_md_files(directory: str, text_splitter):
    """
    Загружает все Markdown-файлы из указанной директории.
    
    Args:
        directory: Путь к директории с Markdown-файлами
        text_splitter: Объект для разбиения текста на фрагменты
        
    Returns:
        Список документов
    """
    documents = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                links = extract_links(content)
                
                # Разбиение текста на фрагменты
                fragments = text_splitter.split_text(content)
                
                for fragment in fragments:
                    # Проверяем, что очищенный от пробелов фрагмент содержит не менее 30 символов
                    if len(fragment.strip()) >= 30:
                        documents.append(
                            Document(
                                page_content=fragment,
                                metadata={
                                    "source": filename,
                                    "links": links
                                }
                            )
                        )
    
    return documents

def load_data_to_chroma(
    data_directory: str = "/Users/sergey/Desktop/Voise_RAG/DATA/articles/cleaned",
    collection_name: str = "knowledge_markdown",
    embedding_model_name: str = "intfloat/multilingual-e5-base"
):
    """
    Загрузка данных в ChromaDB.
    
    Args:
        data_directory: Путь к директории с данными
        collection_name: Имя коллекции в ChromaDB
        embedding_model_name: Название модели для эмбеддингов
    """
    # Получаем абсолютный путь к директории проекта
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Получаем абсолютный путь к базе данных
    persist_directory = os.path.join(project_dir, "chroma_langchain_db", "knowledge")
    
    print(f"Путь к базе данных: {persist_directory}")
    
    # Создаем директорию для базы данных, если она не существует
    os.makedirs(persist_directory, exist_ok=True)
    
    # Загрузка модели эмбеддингов
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    # Создание объекта для разбиения текста с указанными настройками
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["===="],
        chunk_size=2500,
        chunk_overlap=250
    )
    
    # Загрузка векторного хранилища
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    
    # Проверка количества документов перед загрузкой
    collection_count_before = vector_store._collection.count()
    print(f"Количество документов в коллекции {collection_name} перед загрузкой: {collection_count_before}")
    
    # Если в коллекции уже есть документы, спрашиваем пользователя, хочет ли он перезаписать их
    if collection_count_before > 0:
        answer = input("В коллекции уже есть документы. Хотите перезаписать их? (y/n): ")
        if answer.lower() != 'y':
            print("Загрузка отменена.")
            return
        
        # Удаляем существующую коллекцию
        vector_store._collection.delete(where={})
        print(f"Коллекция {collection_name} очищена.")
    
    # Загрузка документов
    documents = load_md_files(data_directory, text_splitter)
    
    print(f"Загружено {len(documents)} документов из {data_directory}")
    
    # Добавление документов в векторное хранилище
    vector_store.add_documents(documents)
    
    # В новой версии langchain_chroma метод persist() не нужен,
    # документы сохраняются автоматически после add_documents()
    
    print(f"Данные успешно загружены в коллекцию {collection_name}")
    
    # Проверка количества документов после загрузки
    collection_count_after = vector_store._collection.count()
    print(f"Количество документов в коллекции {collection_name} после загрузки: {collection_count_after}")

if __name__ == "__main__":
    load_data_to_chroma()