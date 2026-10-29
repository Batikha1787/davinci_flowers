"""
Модуль работы с базой данных SQLite для RAG-системы DaVinci Flowers.
Без сторонних зависимостей (стандартная библиотека sqlite3).
"""

import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "davinci.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # Обязательно включаем поддержку внешних ключей для каскадного удаления
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    """Создаёт таблицы документов и фрагментов, если они ещё не существуют."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                content TEXT NOT NULL,
                uploaded_at TEXT NOT NULL
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fragments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                vector TEXT NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
            );
        """)
        conn.commit()

def add_document(name: str, content: str) -> int:
    """Сохраняет документ (включая исходный текст для возможности переиндексации)."""
    uploaded_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO documents (name, content, uploaded_at) VALUES (?, ?, ?)",
            (name, content, uploaded_at)
        )
        conn.commit()
        return cursor.lastrowid

def delete_document(doc_id: int) -> bool:
    """Удаляет документ. Фрагменты удаляются автоматически благодаря ON DELETE CASCADE."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
        return cursor.rowcount > 0

def get_document(doc_id: int):
    """Возвращает документ по его id."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, content, uploaded_at FROM documents WHERE id = ?", (doc_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def list_documents():
    """Возвращает список документов с количеством фрагментов."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT d.id, d.name, d.uploaded_at, COUNT(f.id) as fragment_count
            FROM documents d
            LEFT JOIN fragments f ON d.id = f.document_id
            GROUP BY d.id
            ORDER BY d.id DESC
        """)
        return [dict(row) for row in cursor.fetchall()]

def save_fragments(doc_id: int, fragments: list):
    """
    Сохраняет фрагменты документа.
    fragments: список кортежей/словарей вида (chunk_index, text, vector_list)
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        # На случай переиндексации сначала удаляем старые фрагменты этого документа
        cursor.execute("DELETE FROM fragments WHERE document_id = ?", (doc_id,))
        for chunk_idx, text, vector in fragments:
            vector_json = json.dumps(vector)
            cursor.execute(
                "INSERT INTO fragments (document_id, chunk_index, text, vector) VALUES (?, ?, ?, ?)",
                (doc_id, chunk_idx, text, vector_json)
            )
        conn.commit()

def get_all_fragments():
    """
    Возвращает все фрагменты вместе с названием документа для поиска похожих.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT f.id, f.document_id, d.name as doc_name, f.chunk_index, f.text, f.vector
            FROM fragments f
            JOIN documents d ON f.document_id = d.id
        """)
        rows = cursor.fetchall()
        result = []
        for row in rows:
            result.append({
                "id": row["id"],
                "document_id": row["document_id"],
                "doc_name": row["doc_name"],
                "chunk_index": row["chunk_index"],
                "text": row["text"],
                "vector": json.loads(row["vector"])
            })
        return result

# Инициализируем БД при импорте
init_db()
