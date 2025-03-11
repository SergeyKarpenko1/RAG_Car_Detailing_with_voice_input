import os
import requests
from dotenv import load_dotenv

def get_openrouter_models():
    """Получает список доступных моделей в OpenRouter."""
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        raise ValueError("API ключ не найден. Укажите его в .env файле или переменной окружения.")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://localhost",  # Требуется для OpenRouter
        "X-Title": "Model Check Script"
    }
    
    response = requests.get("https://openrouter.ai/api/v1/models", headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Ошибка запроса: {response.status_code}, {response.text}")
    
    return response.json()

if __name__ == "__main__":
    try:
        models_data = get_openrouter_models()
        
        # Фильтруем модели, которые могут быть полезны для распознавания речи
        speech_models = []
        for model in models_data.get("data", []):
            model_id = model.get("id", "")
            if "whisper" in model_id.lower() or "speech" in model_id.lower() or "audio" in model_id.lower():
                speech_models.append({
                    "id": model_id,
                    "name": model.get("name", ""),
                    "context_length": model.get("context_length", 0),
                    "pricing": model.get("pricing", {})
                })
        
        print("Модели для распознавания речи:")
        for model in speech_models:
            print(f"ID: {model['id']}")
            print(f"Название: {model['name']}")
            print(f"Контекстное окно: {model['context_length']}")
            
            if model['pricing']:
                input_price = model['pricing'].get('input', 0)
                output_price = model['pricing'].get('output', 0)
                print(f"Цена за ввод: ${input_price} за 1M токенов")
                print(f"Цена за вывод: ${output_price} за 1M токенов")
            
            print("-" * 50)
        
        if not speech_models:
            print("Не найдено моделей для распознавания речи. Вывожу все доступные модели:")
            for model in models_data.get("data", []):
                print(f"ID: {model.get('id', '')}")
                print(f"Название: {model.get('name', '')}")
                print("-" * 50)
    
    except Exception as e:
        print(f"Ошибка: {e}")