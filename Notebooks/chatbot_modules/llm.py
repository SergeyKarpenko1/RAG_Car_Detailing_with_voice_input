import os
from typing import Optional, Callable, Dict, Any
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

class LanguageModelManager:
    """
    Класс для управления языковыми моделями.
    Поддерживает загрузку моделей через OpenRouter.
    """
    
    def __init__(self):
        """Инициализация менеджера языковых моделей."""
        load_dotenv()
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        
        if not self.api_key:
            raise ValueError("API ключ не найден. Укажите его в .env файле или переменной окружения.")
        
        self.loaded_models = {}
    
    def load_model(
        self, 
        model_name: str, 
        temperature: float = 0.7, 
        max_tokens: int = 512,
        model_id: Optional[str] = None
    ) -> Callable[[str], str]:
        """
        Загружает указанную модель и возвращает функцию для запроса.
        
        Args:
            model_name: Название модели в OpenRouter
            temperature: Температура генерации (0.0 - 1.0)
            max_tokens: Максимальное количество токенов в ответе
            model_id: Идентификатор модели для кэширования (если None, используется model_name)
            
        Returns:
            Функция для генерации ответа
        """
        model_id = model_id or model_name
        
        # Проверяем, загружена ли уже модель с такими параметрами
        model_key = f"{model_id}_{temperature}_{max_tokens}"
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]
        
        # Создаем экземпляр модели
        llm = ChatOpenAI(
            openai_api_key=self.api_key,
            openai_api_base=self.base_url,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # Создаем функцию для генерации ответа
        def generate_response(prompt: str) -> str:
            try:
                # Если метод invoke доступен, передаём строку запроса
                if callable(getattr(llm, "invoke", None)):
                    response = llm.invoke(prompt)
                else:
                    response = llm([HumanMessage(content=prompt)])
            except Exception as e:
                return f"Ошибка вызова модели: {e}"
            
            if not response:
                return "Ошибка: не получен ответ от модели."
            
            return response.content
        
        # Сохраняем функцию в кэше
        self.loaded_models[model_key] = generate_response
        
        return generate_response
    
    def get_available_models(self) -> Dict[str, Any]:
        """
        Получение списка доступных моделей в OpenRouter.
        
        Returns:
            Словарь с информацией о доступных моделях
        """
        # Здесь можно добавить запрос к API OpenRouter для получения списка моделей
        # Пока возвращаем статический список популярных моделей
        return {
            "gpt-3.5-turbo": {
                "description": "GPT-3.5 Turbo от OpenAI",
                "context_length": 16385,
                "pricing": "Низкая стоимость"
            },
            "gpt-4": {
                "description": "GPT-4 от OpenAI",
                "context_length": 8192,
                "pricing": "Высокая стоимость"
            },
            "claude-3-opus": {
                "description": "Claude 3 Opus от Anthropic",
                "context_length": 200000,
                "pricing": "Высокая стоимость"
            },
            "claude-3-sonnet": {
                "description": "Claude 3 Sonnet от Anthropic",
                "context_length": 200000,
                "pricing": "Средняя стоимость"
            },
            "claude-3-haiku": {
                "description": "Claude 3 Haiku от Anthropic",
                "context_length": 200000,
                "pricing": "Низкая стоимость"
            },
            "deepseek/deepseek-r1-distill-llama-8b": {
                "description": "DeepSeek R1 Distill Llama 8B",
                "context_length": 4096,
                "pricing": "Низкая стоимость"
            },
            "qwen/qwq-32b": {
                "description": "Qwen QWQ 32B",
                "context_length": 8192,
                "pricing": "Низкая стоимость"
            }
        }