import os
from typing import List, Dict, Any, Tuple, Optional, Type

from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from huggingface_hub import snapshot_download
from pydantic import Field, BaseModel

class DocumentRetriever(BaseRetriever, BaseModel):
    """
    Класс для поиска релевантных документов в базе знаний.
    Использует комбинацию векторного поиска и поиска по ключевым словам (BM25).
    """
    
    persist_directory: str = Field(default="./chroma_langchain_db/knowledge")
    collection_name: str = Field(default="knowledge_markdown")
    embedding_model_name: str = Field(default="intfloat/multilingual-e5-base")
    similarity_top_k: int = Field(default=10)
    bm25_top_k: int = Field(default=10)
    similarity_weight: float = Field(default=0.5)
    bm25_weight: float = Field(default=0.5)
    use_reranker: bool = Field(default=True)
    reranker_model_name: str = Field(default="DiTy/cross-encoder-russian-msmarco")
    reranker_top_k: int = Field(default=5)
    
    embeddings: Any = Field(default=None)
    reranker: Any = Field(default=None)
    vector_store: Any = Field(default=None)
    similarity_retriever: Any = Field(default=None)
    bm25_retriever: Any = Field(default=None)
    ensemble_retriever: Any = Field(default=None)
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        """
        Инициализация ретривера.
        """
        super().__init__(**kwargs)
        self._initialize_components()
    
    def _initialize_components(self):
        """
        Инициализация компонентов ретривера.
        """
        # Загрузка модели эмбеддингов
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        
        # Инициализация реранкера, если он включен
        self.reranker = None
        if self.use_reranker:
            try:
                print(f"Загрузка модели реранкера {self.reranker_model_name}...")
                # Скачивает всю модель с индикатором прогресса
                local_model_path = snapshot_download(self.reranker_model_name)
                print(f"Модель реранкера загружена: {local_model_path}")
                
                # Инициализируем CrossEncoder, используя локальный путь
                self.reranker = CrossEncoder(local_model_path, max_length=512, device='cpu')
                print("Реранкер успешно инициализирован.")
            except Exception as e:
                print(f"Ошибка при инициализации реранкера: {e}")
                print("Реранкер будет отключен.")
                self.use_reranker = False
        
        # Выводим путь к базе данных для отладки
        print(f"Используется путь к базе данных: {os.path.abspath(self.persist_directory)}")
        
        # Проверяем существование директории
        if not os.path.exists(self.persist_directory):
            print(f"Директория {self.persist_directory} не существует!")
            os.makedirs(self.persist_directory, exist_ok=True)
            print(f"Создана директория {self.persist_directory}")
        
        # Загрузка векторного хранилища
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # Создание ретривера по векторной близости
        self.similarity_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.similarity_top_k}
        )
        
        # Извлечение всех документов для BM25 ретривера
        raw_collection = self.vector_store._collection.get()
        docs = []
        
        # Проверка наличия документов
        if "documents" in raw_collection and raw_collection["documents"]:
            for content, meta in zip(raw_collection["documents"], raw_collection["metadatas"]):
                docs.append(Document(page_content=content, metadata=meta))
            
            # Создание BM25 ретривера только если есть документы
            if docs:
                self.bm25_retriever = BM25Retriever.from_documents(docs)
                self.bm25_retriever.k = self.bm25_top_k
                
                # Создание ансамбля ретриверов
                self.ensemble_retriever = EnsembleRetriever(
                    retrievers=[self.similarity_retriever, self.bm25_retriever],
                    weights=[self.similarity_weight, self.bm25_weight]
                )
            else:
                print("Предупреждение: Нет документов для BM25Retriever, используется только векторный поиск")
                self.ensemble_retriever = self.similarity_retriever
        else:
            print("Предупреждение: Нет документов в векторном хранилище, используется только векторный поиск")
            self.ensemble_retriever = self.similarity_retriever
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Внутренний метод для получения релевантных документов (требуется для BaseRetriever).
        
        Args:
            query: Запрос пользователя
            
        Returns:
            Список релевантных документов
        """
        # Получаем документы с помощью ансамбля ретриверов
        docs = self.ensemble_retriever.get_relevant_documents(query)
        
        # Если реранкер не используется или не инициализирован, возвращаем результаты как есть
        if not self.use_reranker or self.reranker is None:
            return docs
        
        # Переранжирование результатов с помощью реранкера
        return self._rerank_documents(query, docs)
    
    def _rerank_documents(self, query: str, docs: List[Document]) -> List[Document]:
        """
        Переранжирование документов с помощью реранкера.
        
        Args:
            query: Запрос пользователя
            docs: Список документов для переранжирования
            
        Returns:
            Переранжированный список документов
        """
        if not docs:
            return []
        
        # Подготовка пар запрос-документ для реранкера
        query_doc_pairs = [(query, doc.page_content) for doc in docs]
        
        # Получение оценок релевантности от реранкера
        scores = self.reranker.predict(query_doc_pairs)
        
        # Создание пар (документ, оценка)
        doc_score_pairs = list(zip(docs, scores))
        
        # Сортировка документов по оценке в порядке убывания
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Ограничение количества документов
        reranked_docs = [doc for doc, _ in doc_score_pairs[:self.reranker_top_k]]
        
        return reranked_docs
    
    def get_context_text(self, query: str) -> str:
        """
        Получение контекстного текста из релевантных документов.
        
        Args:
            query: Запрос пользователя
            
        Returns:
            Строка с контекстным текстом
        """
        relevant_docs = self.get_relevant_documents(query)
        context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
        return context_text
    
    def get_context_with_sources(self, query: str) -> Dict[str, Any]:
        """
        Получение контекстного текста с источниками.
        
        Args:
            query: Запрос пользователя
            
        Returns:
            Словарь с контекстным текстом и источниками
        """
        relevant_docs = self.get_relevant_documents(query)
        context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        sources = []
        for doc in relevant_docs:
            if "source" in doc.metadata:
                sources.append(doc.metadata["source"])
            elif "title" in doc.metadata:
                sources.append(doc.metadata["title"])
        
        return {
            "context": context_text,
            "sources": list(set(sources))  # Удаление дубликатов
        }