import os
import chromadb

def check_chroma_collections(
    persist_directory: str = "./chatbot_modules/chroma_langchain_db/knowledge"
):
    """
    Проверка коллекций в базе данных ChromaDB.
    
    Args:
        persist_directory: Путь к директории с векторной базой данных
    """
    # Создаем клиент ChromaDB
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Получаем список коллекций
    collections = client.list_collections()
    
    print(f"Найдено коллекций: {len(collections)}")
    
    # Проверяем каждую коллекцию
    for i, collection_info in enumerate(collections):
        # В новой версии Chroma collection_info это просто строка с именем коллекции
        collection_name = collection_info
        print(f"\nКоллекция {i+1}:")
        print(f"Имя: {collection_name}")
        
        # Получаем коллекцию по имени
        collection = client.get_collection(name=collection_name)
        
        # Получаем количество документов
        count = collection.count()
        print(f"Количество документов: {count}")
        
        # Получаем первые 3 документа из коллекции
        if count > 0:
            results = collection.get(limit=3)
            print("\nПервые 3 документа:")
            for j in range(min(3, len(results["ids"]))):
                doc_id = results["ids"][j]
                print(f"\nДокумент {j+1} (ID: {doc_id}):")
                if "documents" in results and j < len(results["documents"]):
                    doc_content = results["documents"][j]
                    print(f"Содержание: {doc_content[:200]}..." if len(doc_content) > 200 else f"Содержание: {doc_content}")
                if "metadatas" in results and j < len(results["metadatas"]):
                    print(f"Метаданные: {results['metadatas'][j]}")
                print("-" * 80)

if __name__ == "__main__":
    check_chroma_collections()