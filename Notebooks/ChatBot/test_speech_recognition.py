#!/usr/bin/env python3
"""
Скрипт для тестирования распознавания речи.
Позволяет проверить работу различных распознавателей речи.
"""

import os
import argparse
from dotenv import load_dotenv

# Импортируем модуль распознавания речи
import sys
import os

# Добавляем путь к директории с модулями в sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'chatbot_modules'))
from speech_to_text import SpeechRecognizer

def main():
    """Основная функция для тестирования распознавания речи."""
    # Загрузка переменных окружения
    load_dotenv()
    
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Тестирование распознавания речи")
    parser.add_argument("--recognizer", type=str, default="google", 
                        choices=["google", "whisper", "whisper_api", "sphinx"],
                        help="Тип распознавателя речи")
    parser.add_argument("--language", type=str, default="ru-RU", 
                        help="Язык распознавания речи")
    parser.add_argument("--timeout", type=int, default=5, 
                        help="Таймаут ожидания аудио (в секундах)")
    parser.add_argument("--phrase-time-limit", type=int, default=None, 
                        help="Ограничение времени записи фразы (в секундах)")
    args = parser.parse_args()
    
    # Проверка наличия API ключа для Whisper API
    if args.recognizer == "whisper_api" and not os.getenv("OPENAI_API_KEY"):
        print("ОШИБКА: Для Whisper API требуется API ключ OpenAI.")
        print("Укажите его в .env файле или переменной окружения OPENAI_API_KEY.")
        return
    
    try:
        # Создание распознавателя речи
        recognizer = SpeechRecognizer(
            recognizer_type=args.recognizer,
            language=args.language,
            timeout=args.timeout,
            phrase_time_limit=args.phrase_time_limit
        )
        
        print(f"Распознаватель речи: {args.recognizer}")
        print(f"Язык: {args.language}")
        print("Для выхода нажмите Ctrl+C")
        
        # Бесконечный цикл распознавания речи
        while True:
            print("\nГоворите сейчас...")
            text, confidence = recognizer.recognize_from_microphone()
            
            print(f"Распознанный текст: {text}")
            print(f"Уровень уверенности: {confidence:.2f}")
            
            # Пауза перед следующим распознаванием
            input("Нажмите Enter для продолжения или Ctrl+C для выхода...")
            
    except KeyboardInterrupt:
        print("\nЗавершение работы...")
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()