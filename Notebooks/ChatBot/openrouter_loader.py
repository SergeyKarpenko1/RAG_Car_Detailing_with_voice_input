import os
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv

def load_model(model_name: str, temperature: float = 0.7, max_tokens: int = 512):
    """Загружает указанную модель OpenRouter и возвращает функцию для запроса."""
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = "https://openrouter.ai/api/v1"

    if not api_key:
        raise ValueError("API ключ не найден. Укажите его в .env файле или переменной окружения.")

    llm = ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base=base_url,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    def generate_response(prompt: str):
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

    return generate_response

if __name__ == "__main__":
    model = "qwen/qwq-32b:free"  # Поправил название модели
    prompt = "Какая столица Франции?"

    generate = load_model(model)
    print(generate(prompt))