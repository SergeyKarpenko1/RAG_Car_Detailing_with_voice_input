import os
from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper

# Загружаем переменные окружения из .env
load_dotenv()

# Указываем API-ключ Serper
serper_api_key = os.getenv("SERPER_API_KEY")

# Проверяем, что API-ключ установлен
if not serper_api_key:
    raise ValueError("SERPER_API_KEY не установлен в переменных окружения")

# Инструмент поиска
search_tool = GoogleSerperAPIWrapper()

def search_web(query: str):
    """Поиск информации в интернете через Google Serper"""
    results = search_tool.run(query)
    return results if results else None

# Пример использования
if __name__ == "__main__":
    query = "Кто основал компанию Tesla?"
    result = search_web(query)
    print(result)
