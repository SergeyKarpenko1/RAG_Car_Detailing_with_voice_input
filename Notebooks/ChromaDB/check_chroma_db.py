from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

def check_chroma_db(
    persist_directory: str = "./chroma_langchain_db/knowledge",
    collection_name: str = "knowledge_markdown",
    embedding_model_name: str = "intfloat/multilingual-e5-base"
):
    """
    Проверка содержимого базы данных ChromaDB.
    
    Args:
        persist_directory: Путь к директории с векторной базой данных
        collection_name: Имя коллекции в ChromaDB
        embedding_model_name: Название модели для эмбеддингов
    """
    # Загрузка модели эмбеддингов
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    # Загрузка векторного хранилища
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    
    # Получение количества документов
    collection_count = vector_store._collection.count()
    print(f"Количество документов в коллекции {collection_name}: {collection_count}")
    
    # Получение первых 5 документов
    if collection_count > 0:
        results = vector_store.get(include=['documents', 'metadatas'], limit=5)
        
        print("\nПервые 5 документов:")
        for i, doc in enumerate(results["documents"]):
            print(f"\nДокумент {i+1}:")
            print(f"Содержание: {doc[:200]}...")  # Ограничение вывода
            print(f"Метаданные: {results['metadatas'][i]}")
            print("-" * 80)
    
    # Проверка поиска
    query = "Как убрать ржавчину с кузова автомобиля?"
    print(f"\nПоиск по запросу: '{query}'")
    
    search_results = vector_store.similarity_search(query, k=3)
    
    if search_results:
        print("\nНайденные документы:")
        for i, doc in enumerate(search_results):
            print(f"\nДокумент {i+1}:")
            print(f"Содержание: {doc.page_content[:200]}...")  # Ограничение вывода
            print(f"Метаданные: {doc.metadata}")
            print("-" * 80)
    else:
        print("Документы не найдены.")

if __name__ == "__main__":
    check_chroma_db()