import os
import argparse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

# Импортируем модули из текущей директории
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from retriever import DocumentRetriever
from chat import RAGChatBot

def load_vector_store(
    persist_directory: str = "./chroma_langchain_db/knowledge",
    collection_name: str = "knowledge_markdown",
    embedding_model_name: str = "intfloat/multilingual-e5-base"
):
    """
    Загрузка векторного хранилища.
    
    Args:
        persist_directory: Путь к директории с векторной базой данных
        collection_name: Имя коллекции в ChromaDB
        embedding_model_name: Название модели для эмбеддингов
        
    Returns:
        Объект Chroma
    """
    # Загрузка модели эмбеддингов
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    # Загрузка векторного хранилища
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    
    return vector_store

def create_ensemble_retriever(
    vector_store,
    similarity_top_k: int = 5,
    bm25_top_k: int = 5,
    similarity_weight: float = 0.5,
    bm25_weight: float = 0.5
):
    """
    Создание ансамбля ретриверов.
    
    Args:
        vector_store: Векторное хранилище
        similarity_top_k: Количество документов для поиска по векторной близости
        bm25_top_k: Количество документов для поиска по BM25
        similarity_weight: Вес для результатов векторного поиска
        bm25_weight: Вес для результатов BM25 поиска
        
    Returns:
        Объект EnsembleRetriever
    """
    # Создание ретривера по векторной близости
    similarity_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": similarity_top_k}
    )
    
    # Извлечение всех документов для BM25 ретривера
    raw_collection = vector_store._collection.get()
    docs = []
    
    # Проверка наличия документов
    if "documents" in raw_collection and raw_collection["documents"]:
        for content, meta in zip(raw_collection["documents"], raw_collection["metadatas"]):
            docs.append(Document(page_content=content, metadata=meta))
        
        # Создание BM25 ретривера только если есть документы
        if docs:
            bm25_retriever = BM25Retriever.from_documents(docs)
            bm25_retriever.k = bm25_top_k
            
            # Создание ансамбля ретриверов
            ensemble_retriever = EnsembleRetriever(
                retrievers=[similarity_retriever, bm25_retriever],
                weights=[similarity_weight, bm25_weight]
            )
        else:
            print("Предупреждение: Нет документов для BM25Retriever, используется только векторный поиск")
            ensemble_retriever = similarity_retriever
    else:
        print("Предупреждение: Нет документов в векторном хранилище, используется только векторный поиск")
        ensemble_retriever = similarity_retriever
    
    return ensemble_retriever

def main():
    """Основная функция для запуска чат-бота."""
    # Загрузка переменных окружения
    load_dotenv()
    
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Запуск чат-бота с RAG")
    parser.add_argument("--model", type=str, default="anthropic/claude-3.5-haiku", help="Название модели в OpenRouter")
    parser.add_argument("--temperature", type=float, default=0.5, help="Температура генерации (0.0 - 1.0)")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Максимальное количество токенов в ответе")
    parser.add_argument("--use-web-search", action="store_true", default=True, help="Использовать поиск в интернете")
    parser.add_argument("--cache-file", type=str, default="./response_cache.json", help="Путь к файлу для сохранения кэша")
    parser.add_argument("--db-path", type=str, default="./chat_history.db", help="Путь к файлу базы данных SQLite для хранения истории диалогов")
    parser.add_argument("--no-history", action="store_true", help="Отключить сохранение истории диалогов")
    parser.add_argument("--use-voice", action="store_true", default=True, help="Использовать голосовой ввод")
    parser.add_argument("--speech-recognizer", type=str, default="google", choices=["google", "whisper", "whisper_api", "sphinx"], help="Тип распознавателя речи")
    parser.add_argument("--speech-language", type=str, default="ru-RU", help="Язык распознавания речи")
    
    # Параметры для ретривера и реранкера
    parser.add_argument("--similarity-top-k", type=int, default=10, help="Количество документов для поиска по векторной близости")
    parser.add_argument("--bm25-top-k", type=int, default=10, help="Количество документов для поиска по BM25")
    parser.add_argument("--similarity-weight", type=float, default=0.5, help="Вес для результатов векторного поиска")
    parser.add_argument("--bm25-weight", type=float, default=0.5, help="Вес для результатов BM25 поиска")
    parser.add_argument("--no-reranker", action="store_true", help="Отключить использование реранкера")
    parser.add_argument("--reranker-model", type=str, default="DiTy/cross-encoder-russian-msmarco", help="Название модели для реранкера")
    parser.add_argument("--reranker-top-k", type=int, default=5, help="Количество документов после переранжирования")
    args = parser.parse_args()
    
    # Проверка наличия API ключа
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("API ключ не найден. Укажите его в .env файле или переменной окружения.")
    
    # Создание модели
    llm = ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    
    # Создание ретривера с реранкером
    retriever = DocumentRetriever(
        persist_directory="./chroma_langchain_db/knowledge",
        collection_name="knowledge_markdown",
        embedding_model_name="intfloat/multilingual-e5-base",
        similarity_top_k=args.similarity_top_k,
        bm25_top_k=args.bm25_top_k,
        similarity_weight=args.similarity_weight,
        bm25_weight=args.bm25_weight,
        use_reranker=not args.no_reranker,
        reranker_model_name=args.reranker_model,
        reranker_top_k=args.reranker_top_k
    )
    
    # Создание чат-бота
    chatbot = RAGChatBot(
        llm=llm,
        retriever=retriever,
        cache_file=args.cache_file,
        db_path=None if args.no_history else args.db_path,  # Отключаем сохранение истории, если указан параметр --no-history
        use_web_search=args.use_web_search,
        max_web_search_attempts=3,
        max_relevant_sources=3,
        use_voice_input=args.use_voice,
        speech_recognizer_type=args.speech_recognizer,
        speech_language=args.speech_language
    )
    
    # Запуск интерактивного чата
    chatbot.run_interactive_chat()

if __name__ == "__main__":
    main()