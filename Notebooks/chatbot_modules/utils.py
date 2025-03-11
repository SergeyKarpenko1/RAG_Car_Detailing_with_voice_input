import os
import json
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

class ChatHistory:
    """
    Класс для управления историей диалога.
    Поддерживает сохранение и загрузку истории.
    """
    
    def __init__(self, history_file: Optional[str] = None):
        """
        Инициализация истории диалога.
        
        Args:
            history_file: Путь к файлу для сохранения истории
        """
        self.history: List[Dict[str, str]] = []
        self.history_file = history_file
    
    def add_message(self, role: str, content: str) -> None:
        """
        Добавление сообщения в историю.
        
        Args:
            role: Роль отправителя (user/bot)
            content: Содержание сообщения
        """
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
    
    def get_last_n_messages(self, n: int) -> List[Dict[str, str]]:
        """
        Получение последних N сообщений.
        
        Args:
            n: Количество сообщений
            
        Returns:
            Список последних N сообщений
        """
        return self.history[-n:] if n < len(self.history) else self.history
    
    def format_history_for_prompt(self, n: int = 10) -> str:
        """
        Форматирование истории для промпта.
        
        Args:
            n: Количество последних сообщений
            
        Returns:
            Отформатированная история
        """
        messages = self.get_last_n_messages(n)
        formatted_history = []
        
        for msg in messages:
            role = "Пользователь" if msg["role"] == "user" else "Бот"
            formatted_history.append(f"{role}: {msg['content']}")
        
        return "\n".join(formatted_history)
    
    def save_history(self, file_path: Optional[str] = None) -> None:
        """
        Сохранение истории в файл.
        
        Args:
            file_path: Путь к файлу (если None, используется self.history_file)
        """
        file_path = file_path or self.history_file
        if not file_path:
            raise ValueError("Не указан путь для сохранения истории")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
    
    def load_history(self, file_path: Optional[str] = None) -> None:
        """
        Загрузка истории из файла.
        
        Args:
            file_path: Путь к файлу (если None, используется self.history_file)
        """
        file_path = file_path or self.history_file
        if not file_path:
            raise ValueError("Не указан путь для загрузки истории")
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                self.history = json.load(f)


class ResponseCache:
    """
    Класс для кэширования ответов на вопросы.
    """
    
    def __init__(self, cache_file: Optional[str] = None):
        """
        Инициализация кэша ответов.
        
        Args:
            cache_file: Путь к файлу для сохранения кэша
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_file = cache_file
        
        if cache_file and os.path.exists(cache_file):
            self.load_cache()
    
    def get_response(self, question: str) -> Optional[Dict[str, Any]]:
        """
        Получение ответа из кэша.
        
        Args:
            question: Вопрос пользователя
            
        Returns:
            Кэшированный ответ или None, если ответа нет в кэше
        """
        # Нормализация вопроса (приведение к нижнему регистру, удаление лишних пробелов)
        normalized_question = question.lower().strip()
        
        return self.cache.get(normalized_question)
    
    def add_response(self, question: str, response: str, context: Optional[str] = None, sources: Optional[List[str]] = None, used_web_search: bool = False) -> None:
        """
        Добавление ответа в кэш.
        
        Args:
            question: Вопрос пользователя
            response: Ответ на вопрос
            context: Контекст, использованный для генерации ответа
            sources: Источники, использованные для генерации ответа
            used_web_search: Использовался ли поиск в интернете
        """
        # Нормализация вопроса
        normalized_question = question.lower().strip()
        
        self.cache[normalized_question] = {
            "response": response,
            "context": context,
            "sources": sources,
            "timestamp": time.time(),
            "used_web_search": used_web_search
        }
        
        if self.cache_file:
            self.save_cache()
    
    def save_cache(self, file_path: Optional[str] = None) -> None:
        """
        Сохранение кэша в файл.
        
        Args:
            file_path: Путь к файлу (если None, используется self.cache_file)
        """
        file_path = file_path or self.cache_file
        if not file_path:
            raise ValueError("Не указан путь для сохранения кэша")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
    
    def load_cache(self, file_path: Optional[str] = None) -> None:
        """
        Загрузка кэша из файла.
        
        Args:
            file_path: Путь к файлу (если None, используется self.cache_file)
        """
        file_path = file_path or self.cache_file
        if not file_path:
            raise ValueError("Не указан путь для загрузки кэша")
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
            except json.JSONDecodeError:
                print(f"Ошибка при загрузке кэша из файла {file_path}. Создан новый кэш.")
                self.cache = {}
                # Создаем резервную копию поврежденного файла
                if os.path.exists(file_path):
                    backup_path = f"{file_path}.bak"
                    try:
                        os.rename(file_path, backup_path)
                        print(f"Создана резервная копия поврежденного файла кэша: {backup_path}")
                    except Exception as e:
                        print(f"Не удалось создать резервную копию файла кэша: {e}")


# Интеграция с поиском в интернете (если доступен SERPER_API_KEY)
try:
    from langchain_community.utilities import GoogleSerperAPIWrapper
    
    # Указываем API-ключ Serper
    serper_api_key = os.getenv("SERPER_API_KEY")
    
    if serper_api_key:
        # Инициализация инструмента поиска с явной передачей API ключа
        search_tool = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)
        
        def search_web(query: str) -> Optional[str]:
            """
            Поиск информации в интернете через Google Serper.
            
            Args:
                query: Поисковый запрос
                
            Returns:
                Результаты поиска или None в случае ошибки
            """
            try:
                results = search_tool.run(query)
                return results if results else None
            except Exception as e:
                print(f"Ошибка при поиске в интернете: {e}")
                return None
    else:
        def search_web(query: str) -> Optional[str]:
            """Заглушка для поиска, если API-ключ не установлен."""
            return None
except ImportError:
    def search_web(query: str) -> Optional[str]:
        """Заглушка для поиска, если библиотека не установлена."""
        return None