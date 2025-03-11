from typing import Optional, List, Dict, Any, Union, Tuple
import os
import time
import uuid

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.tools import Tool

from utils import ResponseCache
from speech_to_text import SpeechRecognizer
from db import ChatDatabase

class RAGChatBot:
    """
    Чат-бот на основе Retrieval-Augmented Generation (RAG) с использованием LangChain.
    """
    
    def __init__(
        self,
        llm: ChatOpenAI,
        retriever: BaseRetriever,
        cache_file: Optional[str] = None,
        db_path: Optional[str] = "./chat_history.db",
        use_web_search: bool = False,
        max_web_search_attempts: int = 3,
        max_relevant_sources: int = 3,
        use_voice_input: bool = False,
        speech_recognizer_type: str = "google",
        speech_language: str = "ru-RU"
    ):
        """
        Инициализация чат-бота.
        
        Args:
            llm: Модель для генерации ответов
            retriever: Ретривер для поиска документов
            cache_file: Путь к файлу для сохранения кэша
            db_path: Путь к файлу базы данных SQLite для хранения истории диалогов
            use_web_search: Использовать ли поиск в интернете
            max_web_search_attempts: Максимальное количество попыток поиска в интернете
            max_relevant_sources: Максимальное количество релевантных источников
            use_voice_input: Использовать ли голосовой ввод
            speech_recognizer_type: Тип распознавателя речи ('google', 'whisper', 'whisper_api', 'sphinx')
            speech_language: Язык распознавания речи
        """
        self.llm = llm
        self.retriever = retriever
        self.use_web_search = use_web_search
        self.max_web_search_attempts = max_web_search_attempts
        self.max_relevant_sources = max_relevant_sources
        self.use_voice_input = use_voice_input
        self.speech_recognizer_type = speech_recognizer_type
        self.speech_language = speech_language
        
        # Создание кэша ответов
        self.cache = ResponseCache(cache_file)
        
        # Создание базы данных для хранения истории диалогов
        self.db = ChatDatabase(db_path) if db_path else None
        
        # Создание памяти для хранения истории диалога в оперативной памяти
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Инициализация распознавателя речи, если включен голосовой ввод
        self.speech_recognizer = None
        if use_voice_input:
            try:
                self.speech_recognizer = SpeechRecognizer(
                    recognizer_type=speech_recognizer_type,
                    language=speech_language
                )
                print(f"Голосовой ввод активирован (распознаватель: {speech_recognizer_type})")
            except Exception as e:
                print(f"Ошибка при инициализации распознавателя речи: {e}")
                print("Голосовой ввод будет отключен")
                self.use_voice_input = False
        
        # Создание инструмента для поиска в интернете
        self.search_tool = None
        if use_web_search:
            try:
                serper_api_key = os.getenv("SERPER_API_KEY")
                if serper_api_key:
                    search = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)
                    self.search_tool = Tool(
                        name="Search",
                        func=search.run,
                        description="Поиск информации в интернете"
                    )
            except Exception as e:
                print(f"Ошибка при инициализации инструмента поиска: {e}")
        
        # Создание промптов
        self._create_prompts()
        
        # Создание цепочки для обработки запросов
        self._create_chain()
    
    def _create_prompts(self):
        """Создание промптов для цепочки обработки запросов."""
        # Промпт для комбинирования контекста и вопроса
        self.qa_prompt = ChatPromptTemplate.from_template("""Ты - полезный ассистент, который отвечает на вопросы пользователя о уходе за автомобилем.
ВАЖНО: Используй ТОЛЬКО предоставленную информацию из базы знаний для ответов.
НЕ используй свои предобученные знания или информацию, которую ты получил при обучении.
Если в предоставленной информации нет ответа на вопрос, явно сообщи: "В базе данных нет информации по этому вопросу."
Отвечай кратко, но информативно.
Не выдумывай информацию, которой нет в предоставленных данных.

Контекст:
{context}

Вопрос: {input}

Твой ответ:""")
        
        # Промпт для поиска в интернете
        self.web_search_prompt = ChatPromptTemplate.from_template("""Информация не найдена в базе знаний. Выполняю поиск в интернете...

Результаты поиска в интернете:
{web_results}

ВАЖНО:
1. Используй ТОЛЬКО информацию из результатов поиска.
2. НЕ используй свои предобученные знания.
3. Укажи в своем ответе, что информация получена из интернета.
4. Если в результатах поиска нет ответа на вопрос, явно сообщи: "В результатах поиска нет информации по этому вопросу."

Вопрос: {input}

Твой ответ:""")
        
        # Промпт для комбинирования ответов из базы знаний и интернета
        self.combine_prompt = ChatPromptTemplate.from_template("""Ты - полезный ассистент, который отвечает на вопросы пользователя о уходе за автомобилем.
ВАЖНО: Используй ТОЛЬКО предоставленную информацию для ответов.
НЕ используй свои предобученные знания или информацию, которую ты получил при обучении.
Отвечай кратко, но информативно.

Информация из базы знаний:
{db_answer}

Информация из интернета:
{web_answer}

Если информация из базы знаний достаточна, используй её.
Если информация из базы знаний недостаточна, но есть информация из интернета, используй информацию из интернета и укажи это в ответе.
Если информации недостаточно ни в базе знаний, ни в интернете, сообщи об этом.

Вопрос: {input}

Твой ответ:""")
    
    def _create_chain(self):
        """Создание цепочки для обработки запросов."""
        # Создание цепочки для ответов на основе базы знаний
        self.qa_chain = create_stuff_documents_chain(
            self.llm,
            self.qa_prompt
        )
        
        # Создание цепочки для ответов на основе поиска в интернете
        if self.search_tool:
            self.web_search_chain = self.web_search_prompt | self.llm | StrOutputParser()
        
        # Создание цепочки для комбинирования ответов
        self.combine_chain = self.combine_prompt | self.llm | StrOutputParser()
        
        # Создание основной цепочки для обработки запросов
        self.retrieval_chain = create_retrieval_chain(
            self.retriever,
            self.qa_chain
        )
        
        # Создаем словарь для хранения истории диалога
        self.chat_history = []
    
    def _get_web_search_results(self, query: str) -> Optional[str]:
        """
        Получение результатов поиска в интернете.
        
        Args:
            query: Поисковый запрос
            
        Returns:
            Результаты поиска или None
        """
        if not self.search_tool:
            return None
        
        # Добавляем контекст об уходе за автомобилем в поисковый запрос
        auto_context = "уход за автомобилем"
        enhanced_query = f"{query} {auto_context}"
        
        print(f"Поиск в интернете: {enhanced_query}")
        
        for attempt in range(self.max_web_search_attempts):
            try:
                results = self.search_tool.run(enhanced_query)
                if results and len(results) > 50:  # Минимальная длина полезного результата
                    return results
            except Exception as e:
                print(f"Ошибка при поиске в интернете (попытка {attempt+1}): {e}")
        
        return "Не удалось найти релевантную информацию в интернете."
    
    def answer(self, question: str, user_id: str = "default_user", use_cache: bool = True) -> Dict[str, Any]:
        """
        Генерация ответа на вопрос пользователя.
        
        Args:
            question: Вопрос пользователя
            user_id: Идентификатор пользователя
            use_cache: Использовать ли кэш для ответов
            
        Returns:
            Словарь с ответом и дополнительной информацией
        """
        start_time = time.time()
        
        # Проверка кэша
        if use_cache:
            cached_response = self.cache.get_response(question)
            if cached_response:
                # Если используется база данных, сохраняем сообщения в историю
                if self.db:
                    self.db.add_message(user_id, "user", question)
                    self.db.add_message(user_id, "bot", cached_response["response"])
                
                return {
                    "answer": cached_response["response"],
                    "sources": cached_response.get("sources", []),
                    "from_cache": True,
                    "time_taken": 0,
                    "used_web_search": cached_response.get("used_web_search", False)
                }
        
        # Получение ответа из базы знаний
        db_response = self.retrieval_chain.invoke({
            "input": question,
            "chat_history": self.chat_history
        })
        
        # Обновляем историю диалога в оперативной памяти
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=db_response["answer"]))
        
        # Ограничиваем историю диалога до последних 10 сообщений
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]
        
        # Если используется база данных, сохраняем сообщения в историю
        if self.db:
            self.db.add_message(user_id, "user", question)
            self.db.add_message(user_id, "bot", db_response["answer"])
        
        db_answer = db_response["answer"]
        source_documents = db_response.get("context", [])
        
        # Ограничение количества источников
        sources = []
        for doc in source_documents[:self.max_relevant_sources]:
            if hasattr(doc, 'metadata'):
                if "source" in doc.metadata:
                    sources.append(doc.metadata["source"])
                elif "title" in doc.metadata:
                    sources.append(doc.metadata["title"])
        
        # Проверка, достаточно ли информации в базе знаний
        used_web_search = False
        web_answer = None
        
        # Логирование для отладки
        print(f"\nОтвет из базы знаний: {db_answer}")
        print(f"Длина ответа: {len(db_answer)} символов")
        print(f"Поиск в интернете включен: {self.use_web_search}")
        
        # Проверяем условия для поиска в интернете
        search_conditions = [
            "не найдено" in db_answer.lower(),
            "недостаточно информации" in db_answer.lower(),
            "нет информации" in db_answer.lower(),
            "в базе данных нет информации" in db_answer.lower(),
            len(db_answer) < 100
        ]
        
        print(f"Условия для поиска в интернете: {search_conditions}")
        
        # Если ответ из базы знаний недостаточен и включен поиск в интернете
        if self.use_web_search and any(search_conditions):
            print("Выполняется поиск в интернете...")
            web_results = self._get_web_search_results(question)
            if web_results:
                used_web_search = True
                web_answer = self.web_search_chain.invoke({
                    "web_results": web_results,
                    "input": question
                })
        
        # Комбинирование ответов, если использовался поиск в интернете
        final_answer = db_answer
        if used_web_search and web_answer:
            final_answer = self.combine_chain.invoke({
                "db_answer": db_answer,
                "web_answer": web_answer,
                "input": question
            })
        
        # Сохранение в кэш
        if use_cache:
            self.cache.add_response(
                question=question,
                response=final_answer,
                sources=sources,
                used_web_search=used_web_search
            )
        
        # Расчет времени выполнения
        time_taken = time.time() - start_time
        
        return {
            "answer": final_answer,
            "sources": sources,
            "from_cache": False,
            "time_taken": time_taken,
            "used_web_search": used_web_search
        }
    
    def get_voice_input(self) -> Tuple[str, float]:
        """
        Получение голосового ввода от пользователя.
        
        Returns:
            Кортеж (распознанный текст, уровень уверенности)
        """
        if not self.use_voice_input or not self.speech_recognizer:
            return "Голосовой ввод не активирован", 0.0
        
        try:
            # Обрабатываем возможные ошибки с event loop
            try:
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    # Если event loop не найден, создаем новый
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except Exception as e:
                print(f"Предупреждение при настройке asyncio: {e}")
            
            # Отключаем параллелизм в tokenizers
            import os
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            # Вызываем распознавание речи
            text, confidence = self.speech_recognizer.recognize_from_microphone()
            
            # Проверяем результат
            if not text or text.startswith("Ошибка") or text.startswith("Таймаут"):
                print(f"Проблема с распознаванием: {text}")
                return f"Проблема с распознаванием: {text}", 0.0
            
            print(f"Успешно распознан текст: {text} (уверенность: {confidence})")
            return text, confidence
            
        except Exception as e:
            import traceback
            print(f"Ошибка при распознавании речи: {e}")
            print(traceback.format_exc())
            return f"Ошибка распознавания: {str(e)}", 0.0
    
    def run_interactive_chat(self, user_id: Optional[str] = None):
        """
        Запуск интерактивного чата в консоли.
        
        Args:
            user_id: Идентификатор пользователя. Если None, генерируется новый идентификатор.
        """
        # Генерация идентификатора пользователя, если не указан
        if user_id is None:
            user_id = str(uuid.uuid4())
        
        print(f"Начало чата (ID пользователя: {user_id}). Для выхода введите 'х'.")
        print("Доступные команды:")
        print("  'г' - переключение между текстовым и голосовым вводом")
        print("  'х' - выход из чата")
        print("  'и' - показать историю диалога")
        print("  'о' - очистить историю диалога")
        
        # Загрузка истории диалога из базы данных, если она используется
        if self.db:
            history = self.db.get_chat_history(user_id)
            if history:
                print(f"\nНайдена предыдущая история диалога ({len(history)} сообщений).")
                print("Последние сообщения:")
                
                # Показываем последние 5 сообщений из истории
                last_messages = history[-10:] if len(history) > 10 else history
                for msg in last_messages:
                    role = "Пользователь" if msg["role"] == "user" else "Бот"
                    print(f"{role}: {msg['message']}")
        
        # Спрашиваем пользователя о предпочтительном способе ввода
        voice_mode_active = False
        if self.use_voice_input:
            while True:
                choice = input("\nКак вы хотите взаимодействовать с чат-ботом? (введите номер)\n1. Голосовой ввод\n2. Текстовый ввод\nВаш выбор: ")
                if choice == '1':
                    voice_mode_active = True
                    print("Выбран голосовой режим ввода.")
                    break
                elif choice == '2':
                    voice_mode_active = False
                    print("Выбран текстовый режим ввода.")
                    break
                else:
                    print("Пожалуйста, введите 1 или 2.")
        else:
            print("Голосовой ввод не активирован. Используется текстовый режим.")
        
        while True:
            if voice_mode_active and self.speech_recognizer:
                print("\nГоворите сейчас (или введите 'г' для переключения на текст, 'х' для выхода, 'и' для истории, 'о' для очистки истории)...")
                
                # Проверяем, не ввел ли пользователь команду с клавиатуры
                user_input = ""
                import sys
                import select
                import os
                
                # Проверяем, есть ли ввод с клавиатуры (только для Unix-подобных систем)
                if os.name == 'posix':
                    # Проверяем, есть ли данные для чтения из stdin
                    r, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if r:
                        user_input = sys.stdin.readline().strip()
                
                # Если пользователь ввел команду с клавиатуры
                if user_input:
                    if user_input.strip().lower() == 'х':
                        print("Завершение чата.")
                        break
                    elif user_input.strip().lower() == 'г':
                        voice_mode_active = not voice_mode_active
                        mode_name = "голосовой" if voice_mode_active else "текстовый"
                        print(f"Переключение на {mode_name} режим ввода.")
                        continue
                    elif user_input.strip().lower() == 'и' and self.db:
                        # Показать историю диалога
                        history = self.db.get_chat_history(user_id)
                        if history:
                            print("\nИстория диалога:")
                            for msg in history:
                                role = "Пользователь" if msg["role"] == "user" else "Бот"
                                print(f"{role}: {msg['message']}")
                        else:
                            print("\nИстория диалога пуста.")
                        continue
                    elif user_input.strip().lower() == 'о' and self.db:
                        # Очистить историю диалога
                        self.db.clear_history(user_id)
                        print("\nИстория диалога очищена.")
                        continue
                else:
                    # Если нет ввода с клавиатуры, используем голосовой ввод
                    text, confidence = self.get_voice_input()
                    
                    if not text or text.startswith("Ошибка") or text.startswith("Таймаут"):
                        print(f"Результат распознавания: {text}")
                        continue
                    
                    print(f"\nРаспознано (уверенность: {confidence:.2f}): {text}")
                    
                    # Проверяем, не является ли распознанный текст командой
                    if text.strip().lower() == 'х' or text.strip().lower() == 'икс':
                        print("Завершение чата.")
                        break
                    elif text.strip().lower() == 'г' or text.strip().lower() == 'же':
                        if not self.use_voice_input:
                            print("Голосовой ввод не активирован в настройках.")
                            continue
                        
                        voice_mode_active = not voice_mode_active
                        mode_name = "голосовой" if voice_mode_active else "текстовый"
                        print(f"Переключение на {mode_name} режим ввода.")
                        continue
                    elif (text.strip().lower() == 'и' or text.strip().lower() == 'история') and self.db:
                        # Показать историю диалога
                        history = self.db.get_chat_history(user_id)
                        if history:
                            print("\nИстория диалога:")
                            for msg in history:
                                role = "Пользователь" if msg["role"] == "user" else "Бот"
                                print(f"{role}: {msg['message']}")
                        else:
                            print("\nИстория диалога пуста.")
                        continue
                    elif (text.strip().lower() == 'о' or text.strip().lower() == 'очистить') and self.db:
                        # Очистить историю диалога
                        self.db.clear_history(user_id)
                        print("\nИстория диалога очищена.")
                        continue
                    
                    user_input = text
            else:
                user_input = input("\nПользователь: ")
                
                # Обработка команд в текстовом режиме
                if user_input.strip().lower() == 'х':
                    print("Завершение чата.")
                    break
                elif user_input.strip().lower() == 'г':
                    if not self.use_voice_input:
                        print("Голосовой ввод не активирован в настройках.")
                        continue
                    
                    voice_mode_active = not voice_mode_active
                    mode_name = "голосовой" if voice_mode_active else "текстовый"
                    print(f"Переключение на {mode_name} режим ввода.")
                    continue
                elif user_input.strip().lower() == 'и' and self.db:
                    # Показать историю диалога
                    history = self.db.get_chat_history(user_id)
                    if history:
                        print("\nИстория диалога:")
                        for msg in history:
                            role = "Пользователь" if msg["role"] == "user" else "Бот"
                            print(f"{role}: {msg['message']}")
                    else:
                        print("\nИстория диалога пуста.")
                    continue
                elif user_input.strip().lower() == 'о' and self.db:
                    # Очистить историю диалога
                    self.db.clear_history(user_id)
                    print("\nИстория диалога очищена.")
                    continue
            
            # Обработка запроса
            response = self.answer(user_input, user_id)
            
            print(f"\nБот: {response['answer']}")
            
            # Вывод источников, если они есть
            if response["sources"]:
                print("\nИсточники:")
                for source in response["sources"]:
                    print(f"- {source}")
            
            # Вывод информации о времени выполнения и использовании веб-поиска
            print(f"\n(Время выполнения: {response['time_taken']:.2f} сек.)")
            if response["used_web_search"]:
                print("(Использован поиск в интернете)")
            
            # Спрашиваем пользователя о способе ввода для следующего вопроса
            if self.use_voice_input:
                while True:
                    choice = input("\nКак вы хотите задать следующий вопрос? (введите номер)\n1. Голосовой ввод\n2. Текстовый ввод\nВаш выбор: ")
                    if choice == '1':
                        voice_mode_active = True
                        print("Выбран голосовой режим ввода.")
                        break
                    elif choice == '2':
                        voice_mode_active = False
                        print("Выбран текстовый режим ввода.")
                        break
                    else:
                        print("Пожалуйста, введите 1 или 2.")