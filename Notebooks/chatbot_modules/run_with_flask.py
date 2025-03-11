#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для запуска приложения через Flask вместо Streamlit.
Этот скрипт создает простой веб-интерфейс с использованием Flask,
который взаимодействует с теми же компонентами, что и Streamlit-приложение.
"""

import os
import sys
import json
from flask import Flask, request, jsonify, render_template_string, Response, send_from_directory
from dotenv import load_dotenv

# Устанавливаем переменные окружения для предотвращения проблем с параллелизмом
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Устанавливаем переменные окружения для PyTorch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Добавляем текущую директорию в путь поиска модулей
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Импортируем патч для torch._classes
try:
    import patch_torch
except ImportError:
    print("Предупреждение: не удалось импортировать patch_torch.py")

# Загрузка переменных окружения
load_dotenv()

# Импортируем наши модули
from chat import RAGChatBot
from retriever import DocumentRetriever
from db import ChatDatabase
from speech_to_text import SpeechRecognizer

# Создаем Flask-приложение
app = Flask(__name__)

# Создаем директорию для статических файлов, если она не существует
static_dir = os.path.join(current_dir, 'static')
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

# Глобальные переменные для хранения состояния
chatbot = None
user_id = None
messages = []
speech_recognizer = None  # Глобальный экземпляр распознавателя речи

# HTML-шаблон для веб-интерфейса
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Автомобильный чат-бот</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Roboto', Arial, sans-serif;
            margin: 0;
            padding: 0;
            color: #333;
            position: relative;
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        /* Простой фон */
        .simple-background {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        
        /* Основной контент */
        .content {
            position: relative;
            z-index: 1;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
            margin-bottom: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #4CAF50;
        }
        
        h1 {
            color: #2E7D32;
            margin-bottom: 10px;
        }
        
        .description {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #4CAF50;
        }
        
        .message {
            padding: 15px;
            margin: 15px 0;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
            line-height: 1.5;
        }
        
        .user {
            background-color: #e3f2fd;
            text-align: right;
            border-right: 4px solid #2196F3;
        }
        
        .assistant {
            background-color: #f5f5f5;
            border-left: 4px solid #4CAF50;
        }
        
        /* Стили для списков в сообщениях */
        .assistant ul, .assistant ol {
            padding-left: 20px;
            margin: 10px 0;
        }
        
        .assistant li {
            margin-bottom: 5px;
        }
        
        .list-item {
            margin: 8px 0;
            padding-left: 10px;
            border-left: 3px solid #4CAF50;
        }
        
        .input-container {
            display: flex;
            margin-top: 20px;
            gap: 10px;
        }
        
        #user-input {
            flex-grow: 1;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        
        #user-input:focus {
            outline: none;
            border-color: #4CAF50;
            box-shadow: 0 0 5px rgba(76, 175, 80, 0.3);
        }
        
        button {
            padding: 12px 18px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            transition: background-color 0.3s, transform 0.1s;
        }
        
        button:hover {
            background-color: #45a049;
        }
        
        button:active {
            transform: scale(0.98);
        }
        
        .sources {
            font-size: 0.85em;
            color: #666;
            margin-top: 10px;
            padding-top: 8px;
            border-top: 1px dashed #ddd;
        }
        
        .mic-button {
            background-color: #f44336;
        }
        
        .mic-button:hover {
            background-color: #e53935;
        }
        
        .mic-button.recording {
            background-color: #9e9e9e;
        }
        
        .clear-button {
            background-color: #2196F3;
        }
        
        .clear-button:hover {
            background-color: #1e88e5;
        }
        
        .status {
            font-style: italic;
            color: #666;
            margin-top: 15px;
            text-align: center;
        }
        
        /* Адаптивность для мобильных устройств */
        @media (max-width: 600px) {
            .content {
                padding: 15px;
                margin: 10px;
            }
            
            .input-container {
                flex-direction: column;
            }
            
            button {
                margin-top: 10px;
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <!-- Простой фон вместо видео -->
    <div class="simple-background"></div>
    
    <div class="content">
        <div class="header">
            <h1>🚗 Автомобильный чат-бот</h1>
        </div>
        
        <div class="description">
            <h3>О чат-боте:</h3>
            <p>Этот чат-бот специализируется на вопросах по уходу за автомобилем. Он использует технологию RAG (Retrieval-Augmented Generation), которая сначала ищет информацию в базе знаний, а затем генерирует ответ на основе найденных данных.</p>
            
            <p><strong>Как работает бот:</strong></p>
            <ol>
                <li><strong>Поиск в базе знаний:</strong> При получении вопроса бот сначала ищет релевантную информацию в своей векторной базе данных, используя семантический поиск и алгоритм BM25.</li>
                <li><strong>Ранжирование результатов:</strong> Найденные документы ранжируются по релевантности с помощью специального реранкера, обученного на русскоязычных данных.</li>
                <li><strong>Генерация ответа:</strong> На основе найденной информации бот генерирует подробный и информативный ответ, используя языковую модель Claude 3.5 Haiku.</li>
                <li><strong>Поиск в интернете (при необходимости):</strong> Если в базе знаний недостаточно информации, бот может обратиться к поиску в интернете для дополнения ответа.</li>
            </ol>
            
            <p>Вы можете задавать вопросы о мойке, полировке, химчистке, защите кузова и других аспектах ухода за автомобилем. Чат-бот поддерживает как текстовый, так и голосовой ввод (с помощью кнопки микрофона).</p>
            
            <p><strong>Примеры вопросов:</strong></p>
            <ul>
                <li>Как правильно мыть двигатель автомобиля?</li>
                <li>Какие средства лучше использовать для полировки фар?</li>
                <li>Как защитить кузов автомобиля зимой?</li>
                <li>Как избавиться от запаха в салоне?</li>
            </ul>
        </div>
        
        <div id="messages-container">
            {% for message in messages %}
            <div class="message {{ message.role }}">
                <div class="message-content">{{ message.content | safe }}</div>
                {% if message.sources %}
                <div class="sources">
                    <strong>Источники:</strong>
                    {% for source in message.sources %}
                    <div>- {{ source }}</div>
                    {% endfor %}
                </div>
                {% endif %}
            </div>
            {% endfor %}
        </div>
        
        <div class="input-container">
            <input type="text" id="user-input" placeholder="Например: Как правильно мыть двигатель автомобиля?">
            <button id="send-button">Отправить</button>
            <button id="mic-button" class="mic-button">🎤</button>
            <button id="clear-button" class="clear-button">Очистить</button>
        </div>
        
        <div id="status" class="status"></div>
    </div>
    
    <script>
        let isRecording = false;
        
        document.addEventListener('DOMContentLoaded', function() {
            // Форматирование сообщений при загрузке страницы
            formatMessages();
            
            // Проверка наличия видео-файла
            checkVideoFile();
        });
        
        // Улучшенная функция для форматирования сообщений
        function formatMessages() {
            try {
                const assistantMessages = document.querySelectorAll('.assistant .message-content');
                
                assistantMessages.forEach(message => {
                    // Получаем текст сообщения
                    let content = message.innerHTML;
                    
                    // Проверяем, содержит ли сообщение нумерованный список
                    if (content.match(/\d+\.\s+[^\n]+/)) {
                        // Разбиваем текст на строки
                        const lines = content.split('\n');
                        let formattedContent = '';
                        let inList = false;
                        
                        // Обрабатываем каждую строку
                        for (let i = 0; i < lines.length; i++) {
                            const line = lines[i];
                            
                            // Проверяем, является ли строка пунктом списка
                            if (line.match(/^\d+\.\s+/)) {
                                // Если это первый пункт списка, начинаем список
                                if (!inList) {
                                    formattedContent += '<ol>';
                                    inList = true;
                                }
                                
                                // Добавляем пункт списка
                                formattedContent += '<li>' + line.replace(/^\d+\.\s+/, '') + '</li>';
                            } else if (line.match(/^[*\-•]\s+/)) {
                                // Если это маркированный список
                                if (!inList || inList === 'ol') {
                                    if (inList === 'ol') {
                                        formattedContent += '</ol>';
                                    }
                                    formattedContent += '<ul>';
                                    inList = 'ul';
                                }
                                
                                // Добавляем пункт списка
                                formattedContent += '<li>' + line.replace(/^[*\-•]\s+/, '') + '</li>';
                            } else {
                                // Если это не пункт списка, закрываем список, если он был открыт
                                if (inList === 'ol') {
                                    formattedContent += '</ol>';
                                    inList = false;
                                } else if (inList === 'ul') {
                                    formattedContent += '</ul>';
                                    inList = false;
                                }
                                
                                // Добавляем обычную строку
                                formattedContent += line + '<br>';
                            }
                        }
                        
                        // Закрываем список, если он остался открытым
                        if (inList === 'ol') {
                            formattedContent += '</ol>';
                        } else if (inList === 'ul') {
                            formattedContent += '</ul>';
                        }
                        
                        // Применяем форматирование
                        message.innerHTML = formattedContent;
                    } else {
                        // Если нет списков, просто заменяем переносы строк на <br>
                        message.innerHTML = content.replace(/\n/g, '<br>');
                    }
                });
            } catch (error) {
                console.error('Ошибка при форматировании сообщений:', error);
            }
        }
        
        // Пустая функция, так как мы убрали видео-фон
        function checkVideoFile() {
            // Ничего не делаем, так как видео-фон убран
            console.log('Видео-фон отключен');
        }
        
        document.getElementById('send-button').addEventListener('click', sendMessage);
        document.getElementById('user-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        document.getElementById('mic-button').addEventListener('click', toggleRecording);
        document.getElementById('clear-button').addEventListener('click', clearHistory);
        
        function sendMessage() {
            const userInput = document.getElementById('user-input').value.trim();
            if (userInput) {
                document.getElementById('status').textContent = 'Отправка сообщения...';
                
                fetch('/send_message', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: userInput }),
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('user-input').value = '';
                    document.getElementById('status').textContent = '';
                    window.location.reload();
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('status').textContent = 'Ошибка при отправке сообщения';
                });
            }
        }
        
        function toggleRecording() {
            const micButton = document.getElementById('mic-button');
            
            // Изменяем вид кнопки и статус
            micButton.textContent = '⏹️';
            micButton.classList.add('recording');
            document.getElementById('status').textContent = 'Запись... Говорите сейчас (до 20 секунд). Нажмите на ⏹️ для остановки.';
            micButton.disabled = true; // Отключаем кнопку на время записи
            
            // Отправляем запрос на запись и распознавание речи
            fetch('/record_audio', {
                method: 'POST',
            })
            .then(response => response.json())
            .then(data => {
                // Возвращаем кнопку в исходное состояние
                micButton.textContent = '🎤';
                micButton.classList.remove('recording');
                micButton.disabled = false;
                
                if (data.success && data.text) {
                    document.getElementById('user-input').value = data.text;
                    document.getElementById('status').textContent = 'Распознано: ' + data.text;
                } else {
                    document.getElementById('status').textContent = 'Не удалось распознать речь: ' + (data.error || 'неизвестная ошибка');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                micButton.textContent = '🎤';
                micButton.classList.remove('recording');
                micButton.disabled = false;
                document.getElementById('status').textContent = 'Ошибка при записи или распознавании речи';
            });
        }
        
        function clearHistory() {
            if (confirm('Вы уверены, что хотите очистить историю диалога?')) {
                fetch('/clear_history', {
                    method: 'POST',
                })
                .then(response => response.json())
                .then(data => {
                    window.location.reload();
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('status').textContent = 'Ошибка при очистке истории';
                });
            }
        }
    </script>
</body>
</html>
"""

# Инициализация компонентов
def initialize_components():
    global chatbot, user_id, messages, speech_recognizer
    
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
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Инициализируем распознаватель речи заранее
        speech_recognizer = SpeechRecognizer(
            recognizer_type="google",
            language="ru-RU",
            timeout=10,
            phrase_time_limit=20  # Увеличиваем время записи до 20 секунд
        )
        print("Глобальный распознаватель речи инициализирован")
        
        # Проверка наличия API ключа
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("API ключ не найден. Укажите его в .env файле или переменной окружения.")
            sys.exit(1)
        
        # Создание модели
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            model_name="anthropic/claude-3.5-haiku",
            temperature=0.5,
            max_tokens=2048,
        )
        
        # Путь к файлу базы данных SQLite для хранения истории диалогов
        db_path = "./chat_history.db"
        
        # Создание ретривера с реранкером
        try:
            retriever = DocumentRetriever(
                persist_directory="./chroma_langchain_db/knowledge",
                collection_name="knowledge_markdown",
                embedding_model_name="intfloat/multilingual-e5-base",
                similarity_top_k=10,
                bm25_top_k=10,
                similarity_weight=0.5,
                bm25_weight=0.5,
                use_reranker=True,
                reranker_model_name="DiTy/cross-encoder-russian-msmarco",
                reranker_top_k=5
            )
        except Exception as e:
            print(f"Ошибка при инициализации ретривера: {e}")
            # Создаем упрощенный ретривер без реранкера
            retriever = DocumentRetriever(
                persist_directory="./chroma_langchain_db/knowledge",
                collection_name="knowledge_markdown",
                embedding_model_name="intfloat/multilingual-e5-base",
                use_reranker=False
            )
        
        # Создание чат-бота
        chatbot = RAGChatBot(
            llm=llm,
            retriever=retriever,
            cache_file="./response_cache.json",
            db_path=db_path,
            use_web_search=True,
            max_web_search_attempts=3,
            max_relevant_sources=3,
            use_voice_input=True,
            speech_recognizer_type="google",
            speech_language="ru-RU"
        )
        
        # Генерация идентификатора пользователя
        import uuid
        user_id = str(uuid.uuid4())
        
        # Загрузка истории диалога
        db = ChatDatabase(db_path)
        history = db.get_chat_history(user_id)
        
        # Обновляем сообщения
        messages = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["message"], "sources": []})
        
        print("Компоненты успешно инициализированы")
    except Exception as e:
        import traceback
        print(f"Ошибка при инициализации компонентов: {e}")
        print(traceback.format_exc())
        sys.exit(1)

# Маршруты Flask-приложения
@app.route('/')
def index():
    import time
    timestamp = int(time.time())  # Добавляем временную метку для предотвращения кэширования
    return render_template_string(HTML_TEMPLATE, messages=messages, timestamp=timestamp)

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.json
    user_message = data.get('message', '')
    
    if user_message:
        try:
            # Добавляем сообщение пользователя
            messages.append({"role": "user", "content": user_message, "sources": []})
            
            # Получаем ответ от чат-бота
            response = chatbot.answer(user_message, user_id=user_id)
            
            # Добавляем ответ бота
            messages.append({
                "role": "assistant",
                "content": response["answer"],
                "sources": response.get("sources", [])
            })
            
            return jsonify({"success": True})
        except Exception as e:
            import traceback
            print(f"Ошибка при обработке сообщения: {e}")
            print(traceback.format_exc())
            
            # Добавляем сообщение об ошибке
            error_message = "Извините, произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте еще раз."
            messages.append({"role": "assistant", "content": error_message, "sources": []})
            
            return jsonify({"success": True})
    
    return jsonify({"success": False, "error": "Пустое сообщение"})

@app.route('/record_audio', methods=['POST'])
def record_audio():
    """
    Единый эндпоинт для записи и распознавания речи.
    При вызове этого эндпоинта сразу начинается запись с микрофона.
    """
    global speech_recognizer
    
    try:
        # Проверяем, что распознаватель речи инициализирован
        if speech_recognizer is None:
            print("Предупреждение: распознаватель речи не был инициализирован, создаем новый")
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
            
            # Создаем распознаватель речи с увеличенным таймаутом
            speech_recognizer = SpeechRecognizer(
                recognizer_type="google",
                language="ru-RU",
                timeout=10,
                phrase_time_limit=20  # Увеличиваем время записи до 20 секунд
            )
        
        print("Начинаю запись с микрофона...")
        # Получаем голосовой ввод, используя глобальный экземпляр
        text, confidence = speech_recognizer.recognize_from_microphone()
        
        print(f"Результат распознавания: {text}, уверенность: {confidence}")
        
        if text and not text.startswith("Ошибка") and not text.startswith("Таймаут"):
            return jsonify({"success": True, "text": text, "confidence": confidence})
        else:
            return jsonify({"success": False, "error": text})
    except Exception as e:
        import traceback
        print(f"Ошибка при записи и распознавании: {e}")
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    try:
        # Очищаем историю диалога
        db = ChatDatabase("./chat_history.db")
        db.clear_history(user_id)
        
        # Очищаем сообщения
        global messages
        messages = []
        
        return jsonify({"success": True})
    except Exception as e:
        import traceback
        print(f"Ошибка при очистке истории: {e}")
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)})

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Обслуживание статических файлов, включая видео-фон."""
    return send_from_directory(static_dir, filename)

# Запуск приложения
if __name__ == '__main__':
    print("Инициализация компонентов...")
    initialize_components()
    
    print("Запуск Flask-приложения...")
    app.run(debug=True, port=8501)