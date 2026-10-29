"""
HTTP-сервер для DaVinci Flowers:
- Раздача статики сайта (index.html, стили, скрипты)
- Защищённый эндпоинт админки /admin
- API чат-бота /api/chat с ограничением запросов (rate limiter)
- API управления документами (загрузка, удаление, переиндексация)
- Проверка размера тела запроса (до 10 МБ, чтобы исключить ошибку 413)
- Безопасная аутентификация по ADMIN_PASSWORD
"""

import os
import re
import json
import time
import secrets
from collections import defaultdict
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse

import db
import rag

# Загружаем переменные окружения
rag.load_env()

PORT = int(os.environ.get("PORT", "8000"))
MAX_BODY_SIZE = 10 * 1024 * 1024  # 10 МБ (с большим запасом для длинных текстов)
FILE_SIZE_LIMIT = 1 * 1024 * 1024 # 1 МБ лимит для файла документов

# Сессии администраторов (токены в памяти)
admin_tokens = set()

# Rate limiter: хранилище времени запросов по IP (не более 30 запросов в минуту)
ip_rate_limits = defaultdict(list)
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX_REQUESTS = 30

def clean_filename(filename: str) -> str:
    """Очищает имя файла от путей и спецсимволов, сохраняя кириллицу и букву 'ё'."""
    filename = os.path.basename(filename)
    cleaned = re.sub(r'[^a-zA-Z0-9а-яА-ЯёЁ\s\.\-_]', '', filename)
    cleaned = cleaned.strip()
    return cleaned or "document.txt"

def is_rate_limited(ip: str) -> bool:
    """Проверяет ограничение числа запросов для IP."""
    now = time.time()
    timestamps = ip_rate_limits[ip]
    # Очищаем старые метки времени
    ip_rate_limits[ip] = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]
    if len(ip_rate_limits[ip]) >= RATE_LIMIT_MAX_REQUESTS:
        return True
    ip_rate_limits[ip].append(now)
    return False

class DaVinciHandler(SimpleHTTPRequestHandler):
    """Кастомный обработчик запросов для сайта DaVinci Flowers."""

    def log_message(self, format, *args):
        # Безопасное логирование: пишем метод и статус, не логируя тело запросов и ключи
        print(f"[{self.log_date_time_string()}] {self.address_string()} - {format % args}")

    def send_json(self, status: int, data: dict):
        """Вспомогательный метод для отправки JSON-ответа."""
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.end_headers()
        self.wfile.write(payload)

    def is_admin_authorized(self) -> bool:
        """Проверяет наличие валидного токена администратора."""
        auth_header = self.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            return token in admin_tokens
        return False

    def do_OPTIONS(self):
        """Поддержка CORS preflight-запросов."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
    def do_HEAD(self):
        return self.do_GET()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # 1. Роут админки
        if path in ("/admin", "/admin/"):
            admin_path = os.path.join(os.path.dirname(__file__), "admin.html")
            try:
                with open(admin_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            except Exception as e:
                return self.send_json(500, {"error": f"Не удалось прочитать admin.html: {e}"})

        # 2. Роут списка документов (только для админа)
        if path == "/api/admin/documents":
            if not self.is_admin_authorized():
                return self.send_json(401, {"error": "Требуется авторизация администратора"})
            docs = db.list_documents()
            return self.send_json(200, {"documents": docs})

        # 3. Статические файлы по умолчанию (index.html и др.)
        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # Проверка размера тела запроса (предотвращение ошибки 413)
        try:
            content_length = int(self.headers.get("Content-Length", 0))
        except (ValueError, TypeError):
            content_length = 0

        if content_length > MAX_BODY_SIZE:
            return self.send_json(413, {"error": "Размер запроса превышает допустимый лимит (10 МБ)"})

        body = self.rfile.read(content_length)

        # 1. ЧАТ-БОТ: /api/chat
        if path == "/api/chat":
            client_ip = self.client_address[0]
            if is_rate_limited(client_ip):
                return self.send_json(429, {"error": "Слишком много запросов. Пожалуйста, подождите немного перед следующим вопросом."})

            try:
                data = json.loads(body.decode("utf-8"))
                query = data.get("message", "")
                threshold = data.get("threshold", None)
                if threshold is not None:
                    threshold = float(threshold)
            except Exception:
                return self.send_json(400, {"error": "Неверный формат JSON в запросе"})

            try:
                result = rag.answer_query(query, custom_threshold=threshold)
                return self.send_json(200, result)
            except Exception as e:
                return self.send_json(500, {
                    "answer": "Произошла временная ошибка при обработке запроса. Пожалуйста, попробуйте снова через минуту.",
                    "error": str(e)
                })

        # 2. АВТОРИЗАЦИЯ АДМИНА: /api/admin/login
        if path == "/api/admin/login":
            try:
                data = json.loads(body.decode("utf-8"))
                password = data.get("password", "")
            except Exception:
                return self.send_json(400, {"error": "Неверный формат JSON"})

            expected_password = os.environ.get("ADMIN_PASSWORD", "davinci2026")
            if password == expected_password:
                token = secrets.token_hex(24)
                admin_tokens.add(token)
                return self.send_json(200, {"token": token, "success": True})
            else:
                return self.send_json(403, {"error": "Неверный пароль администратора"})

        # Проверка прав для всех остальных методов админки
        if path.startswith("/api/admin/"):
            if not self.is_admin_authorized():
                return self.send_json(401, {"error": "Требуется авторизация администратора"})

        # 3. ЗАГРУЗКА ДОКУМЕНТА: /api/admin/upload
        if path == "/api/admin/upload":
            try:
                data = json.loads(body.decode("utf-8"))
                raw_filename = data.get("filename", "document.txt")
                content = data.get("content", "")
            except Exception:
                return self.send_json(400, {"error": "Неверный формат JSON"})

            # Очищаем имя файла (сохраняя русские буквы и 'ё')
            filename = clean_filename(raw_filename)
            ext = os.path.splitext(filename)[1].lower()

            if ext not in [".txt", ".md"]:
                return self.send_json(400, {"error": "Поддерживаются только файлы с расширениями .txt и .md"})

            if len(content.encode("utf-8")) > FILE_SIZE_LIMIT:
                return self.send_json(400, {"error": "Размер файла превышает лимит в 1 МБ"})

            if not content.strip():
                return self.send_json(400, {"error": "Документ не содержит текста"})

            try:
                # 1. Сохраняем документ в БД
                doc_id = db.add_document(filename, content)
                # 2. Индексируем документ (нарезка на фрагменты + эмбеддинги)
                chunk_count = rag.index_document(doc_id)
                return self.send_json(200, {
                    "success": True,
                    "document_id": doc_id,
                    "filename": filename,
                    "fragment_count": chunk_count
                })
            except Exception as e:
                return self.send_json(500, {"error": f"Ошибка сохранения/индексации: {e}"})

        # 4. ПЕРЕИНДЕКСАЦИЯ: /api/admin/reindex
        if path == "/api/admin/reindex":
            try:
                res = rag.reindex_all_documents()
                return self.send_json(200, {"success": True, **res})
            except Exception as e:
                return self.send_json(500, {"error": f"Ошибка переиндексации: {e}"})

        return self.send_json(404, {"error": "Метод не найден"})

    def do_DELETE(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path.startswith("/api/admin/documents/"):
            if not self.is_admin_authorized():
                return self.send_json(401, {"error": "Требуется авторизация администратора"})

            try:
                doc_id = int(path.split("/")[-1])
            except ValueError:
                return self.send_json(400, {"error": "Некорректный ID документа"})

            deleted = db.delete_document(doc_id)
            if deleted:
                return self.send_json(200, {"success": True, "message": f"Документ #{doc_id} и его фрагменты удалены"})
            else:
                return self.send_json(404, {"error": "Документ не найден"})

        return self.send_json(404, {"error": "Метод не найден"})

def run_server():
    """Запуск сервера."""
    server_address = ("", PORT)
    httpd = HTTPServer(server_address, DaVinciHandler)
    print(f"Сервер DaVinci Flowers успешно запущен на порту {PORT}: http://localhost:{PORT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановка сервера...")
        httpd.server_close()

if __name__ == "__main__":
    run_server()
