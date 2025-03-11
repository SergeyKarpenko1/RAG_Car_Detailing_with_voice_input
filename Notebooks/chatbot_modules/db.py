import os
import sqlite3
import time
from typing import List, Dict, Any, Optional, Tuple

class ChatDatabase:
    """
    Класс для работы с базой данных SQLite для хранения истории диалогов.
    """
    
    def __init__(self, db_path: str = "./chat_history.db"):
        """
        Инициализация базы данных.
        
        Args:
            db_path: Путь к файлу базы данных SQLite
        """
        self.db_path = db_path
        self._create_tables()
    
    def _create_tables(self) -> None:
        """Создание необходимых таблиц в базе данных, если они не существуют."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Создание таблицы для хранения истории диалогов
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            role TEXT NOT NULL,
            message_text TEXT NOT NULL
        )
        ''')
        
        # Создание индекса для быстрого поиска по user_id
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_user_id ON chat_history(user_id)
        ''')
        
        conn.commit()
        conn.close()
    
    def add_message(self, user_id: str, role: str, message_text: str) -> None:
        """
        Добавление сообщения в историю диалога.
        
        Args:
            user_id: Идентификатор пользователя
            role: Роль отправителя (user/bot)
            message_text: Текст сообщения
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = time.time()
        
        cursor.execute(
            "INSERT INTO chat_history (user_id, timestamp, role, message_text) VALUES (?, ?, ?, ?)",
            (user_id, timestamp, role, message_text)
        )
        
        conn.commit()
        conn.close()
    
    def get_chat_history(self, user_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Получение истории диалога для конкретного пользователя.
        
        Args:
            user_id: Идентификатор пользователя
            limit: Ограничение количества сообщений (если None, возвращаются все сообщения)
            
        Returns:
            Список сообщений в формате [{"role": "user/bot", "message": "текст сообщения", "timestamp": время}]
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Для доступа к столбцам по имени
        cursor = conn.cursor()
        
        if limit:
            cursor.execute(
                "SELECT * FROM chat_history WHERE user_id = ? ORDER BY timestamp ASC LIMIT ?",
                (user_id, limit)
            )
        else:
            cursor.execute(
                "SELECT * FROM chat_history WHERE user_id = ? ORDER BY timestamp ASC",
                (user_id,)
            )
        
        rows = cursor.fetchall()
        
        history = []
        for row in rows:
            history.append({
                "role": row["role"],
                "message": row["message_text"],
                "timestamp": row["timestamp"]
            })
        
        conn.close()
        return history
    
    def get_last_n_messages(self, user_id: str, n: int) -> List[Dict[str, Any]]:
        """
        Получение последних N сообщений для конкретного пользователя.
        
        Args:
            user_id: Идентификатор пользователя
            n: Количество последних сообщений
            
        Returns:
            Список последних N сообщений
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM chat_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
            (user_id, n)
        )
        
        rows = cursor.fetchall()
        
        # Переворачиваем список, чтобы сообщения были в хронологическом порядке
        rows.reverse()
        
        history = []
        for row in rows:
            history.append({
                "role": row["role"],
                "message": row["message_text"],
                "timestamp": row["timestamp"]
            })
        
        conn.close()
        return history
    
    def format_history_for_prompt(self, user_id: str, n: int = 10) -> str:
        """
        Форматирование истории для промпта.
        
        Args:
            user_id: Идентификатор пользователя
            n: Количество последних сообщений
            
        Returns:
            Отформатированная история
        """
        messages = self.get_last_n_messages(user_id, n)
        formatted_history = []
        
        for msg in messages:
            role = "Пользователь" if msg["role"] == "user" else "Бот"
            formatted_history.append(f"{role}: {msg['message']}")
        
        return "\n".join(formatted_history)
    
    def clear_history(self, user_id: str) -> None:
        """
        Очистка истории диалога для конкретного пользователя.
        
        Args:
            user_id: Идентификатор пользователя
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "DELETE FROM chat_history WHERE user_id = ?",
            (user_id,)
        )
        
        conn.commit()
        conn.close()
    
    def get_all_users(self) -> List[str]:
        """
        Получение списка всех пользователей.
        
        Returns:
            Список идентификаторов пользователей
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT DISTINCT user_id FROM chat_history")
        
        users = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return users