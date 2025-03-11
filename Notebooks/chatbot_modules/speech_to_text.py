# Отключаем параллелизм в tokenizers для предотвращения дедлоков
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import tempfile
import speech_recognition as sr
from typing import Optional, Dict, Any, List, Tuple
from dotenv import load_dotenv
import traceback

class SpeechRecognizer:
    """
    Класс для распознавания речи с использованием различных API.
    """
    
    def __init__(
        self,
        recognizer_type: str = "google",
        language: str = "ru-RU",
        api_key: Optional[str] = None,
        timeout: int = 5,
        phrase_time_limit: Optional[int] = None
    ):
        """
        Инициализация распознавателя речи.
        
        Args:
            recognizer_type: Тип распознавателя ('google', 'whisper', 'vosk', 'sphinx')
            language: Язык распознавания
            api_key: API ключ (если требуется)
            timeout: Таймаут ожидания аудио (в секундах)
            phrase_time_limit: Ограничение времени записи фразы (в секундах)
        """
        try:
            self.recognizer = sr.Recognizer()
            self.recognizer_type = recognizer_type.lower()
            self.language = language
            self.api_key = api_key
            self.timeout = timeout
            self.phrase_time_limit = phrase_time_limit
            
            # Загрузка переменных окружения для API ключей
            load_dotenv()
            
            # Настройка распознавателя в зависимости от типа
            if self.recognizer_type == "whisper_api" and not self.api_key:
                self.api_key = os.getenv("OPENAI_API_KEY")
            
            # Проверка наличия необходимых API ключей
            if self.recognizer_type == "whisper_api" and not self.api_key:
                raise ValueError("Для Whisper API требуется API ключ OpenAI")
        except Exception as e:
            print(f"Ошибка при инициализации распознавателя речи: {e}")
            print(traceback.format_exc())
            raise
    
    def recognize_from_microphone(self) -> Tuple[str, float]:
        """
        Запись и распознавание речи с микрофона.
        
        Returns:
            Кортеж (распознанный текст, уровень уверенности)
        """
        try:
            # Проверяем, доступен ли микрофон
            try:
                microphone = sr.Microphone()
            except Exception as e:
                print(f"Ошибка при инициализации микрофона: {e}")
                print(traceback.format_exc())
                return f"Ошибка при инициализации микрофона: {str(e)}", 0.0
            
            with microphone as source:
                print("Настройка микрофона под окружающий шум...")
                try:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                except Exception as e:
                    print(f"Ошибка при настройке микрофона: {e}")
                    print(traceback.format_exc())
                    return f"Ошибка при настройке микрофона: {str(e)}", 0.0
                
                print("Говорите сейчас...")
                try:
                    audio = self.recognizer.listen(
                        source,
                        timeout=self.timeout,
                        phrase_time_limit=self.phrase_time_limit
                    )
                    print("Запись завершена, распознаю...")
                    
                    return self._process_audio(audio)
                
                except sr.WaitTimeoutError:
                    return "Таймаут: не обнаружено речи", 0.0
                except Exception as e:
                    print(f"Ошибка при записи аудио: {e}")
                    print(traceback.format_exc())
                    return f"Ошибка при записи аудио: {str(e)}", 0.0
        except Exception as e:
            print(f"Неожиданная ошибка при распознавании речи: {e}")
            print(traceback.format_exc())
            return f"Неожиданная ошибка при распознавании речи: {str(e)}", 0.0
    
    def recognize_from_file(self, file_path: str) -> Tuple[str, float]:
        """
        Распознавание речи из аудиофайла.
        
        Args:
            file_path: Путь к аудиофайлу
            
        Returns:
            Кортеж (распознанный текст, уровень уверенности)
        """
        try:
            with sr.AudioFile(file_path) as source:
                audio = self.recognizer.record(source)
                return self._process_audio(audio)
        except Exception as e:
            return f"Ошибка при чтении или распознавании файла: {str(e)}", 0.0
    
    def _process_audio(self, audio: sr.AudioData) -> Tuple[str, float]:
        """
        Обработка аудиоданных с помощью выбранного распознавателя.
        
        Args:
            audio: Аудиоданные для распознавания
            
        Returns:
            Кортеж (распознанный текст, уровень уверенности)
        """
        try:
            if self.recognizer_type == "google":
                # Google Speech Recognition (бесплатно с ограничениями)
                try:
                    result = self.recognizer.recognize_google(
                        audio,
                        language=self.language,
                        show_all=True
                    )
                    
                    if result and isinstance(result, dict) and "alternative" in result:
                        text = result["alternative"][0]["transcript"]
                        confidence = result["alternative"][0].get("confidence", 0.0)
                        return text, confidence
                    elif result and isinstance(result, list) and len(result) > 0:
                        return result[0], 0.0
                    else:
                        return "", 0.0
                except Exception as e:
                    print(f"Ошибка при распознавании с Google: {e}")
                    print(traceback.format_exc())
                    return f"Ошибка при распознавании с Google: {str(e)}", 0.0
            
            elif self.recognizer_type == "whisper_api":
                # OpenAI Whisper API (платно)
                try:
                    result = self.recognizer.recognize_whisper_api(
                        audio,
                        api_key=self.api_key
                    )
                    return result, 1.0  # Whisper API не возвращает уровень уверенности
                except Exception as e:
                    print(f"Ошибка при распознавании с Whisper API: {e}")
                    print(traceback.format_exc())
                    return f"Ошибка при распознавании с Whisper API: {str(e)}", 0.0
            
            elif self.recognizer_type == "whisper":
                # Локальная модель Whisper (требует установки whisper)
                try:
                    import whisper
                except ImportError:
                    return "Ошибка: библиотека whisper не установлена", 0.0
                
                # Сохраняем аудио во временный файл
                temp_filename = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        temp_filename = temp_file.name
                        with open(temp_filename, "wb") as f:
                            f.write(audio.get_wav_data())
                    
                    # Загружаем модель Whisper (small - хороший баланс между качеством и скоростью)
                    model = whisper.load_model("small")
                    result = model.transcribe(temp_filename, language=self.language[:2])
                    return result["text"], result.get("confidence", 1.0)
                except Exception as e:
                    print(f"Ошибка при распознавании с Whisper: {e}")
                    print(traceback.format_exc())
                    return f"Ошибка при распознавании с Whisper: {str(e)}", 0.0
                finally:
                    # Удаляем временный файл в любом случае
                    if temp_filename and os.path.exists(temp_filename):
                        try:
                            os.unlink(temp_filename)
                        except Exception as e:
                            print(f"Ошибка при удалении временного файла: {e}")
            
            elif self.recognizer_type == "sphinx":
                # CMU Sphinx (локальное распознавание, не требует интернета)
                try:
                    result = self.recognizer.recognize_sphinx(audio, language=self.language)
                    return result, 0.8  # Sphinx не возвращает уровень уверенности
                except Exception as e:
                    print(f"Ошибка при распознавании с Sphinx: {e}")
                    print(traceback.format_exc())
                    return f"Ошибка при распознавании с Sphinx: {str(e)}", 0.0
            
            else:
                return f"Неподдерживаемый тип распознавателя: {self.recognizer_type}", 0.0
        
        except sr.UnknownValueError:
            return "Речь не распознана", 0.0
        except sr.RequestError as e:
            print(f"Ошибка сервиса распознавания: {e}")
            return f"Ошибка сервиса распознавания: {str(e)}", 0.0
        except Exception as e:
            print(f"Неожиданная ошибка при обработке аудио: {e}")
            print(traceback.format_exc())
            return f"Неожиданная ошибка при обработке аудио: {str(e)}", 0.0

# Пример использования
if __name__ == "__main__":
    # Создание распознавателя с Google Speech Recognition
    recognizer = SpeechRecognizer(recognizer_type="google", language="ru-RU")
    
    # Распознавание с микрофона
    text, confidence = recognizer.recognize_from_microphone()
    
    print(f"Распознанный текст: {text}")
    print(f"Уровень уверенности: {confidence}")