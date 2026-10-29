"""
Скрипт проверки и оценки качества RAG-системы DaVinci Flowers.
Читает eval/questions.json, запускает проверку через RAG и выводит:
- Вопрос
- Реальное значение сходства (Similarity) и порог
- Ответ бота
- Названия источников
- Результат теста (Успех / Ловушка сработала корректно)
"""

import os
import sys
import json

# Добавляем родительскую директорию в sys.path для импорта rag и db
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import db
import rag

def setup_sample_docs():
    """Загружает тестовые документы из sample_docs, если база пуста."""
    docs = db.list_documents()
    sample_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sample_docs"))
    
    if not os.path.exists(sample_dir):
        print(f"Папка {sample_dir} не найдена.")
        return

    # Загружаем документы из sample_docs
    for filename in sorted(os.listdir(sample_dir)):
        if filename.endswith(".md") or filename.endswith(".txt"):
            filepath = os.path.join(sample_dir, filename)
            # Проверяем, есть ли уже такой документ
            existing = [d for d in docs if d["name"] == filename]
            if not existing:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                doc_id = db.add_document(filename, content)
                print(f"Загружен тестовый документ: {filename} (ID #{doc_id})")
                try:
                    count = rag.index_document(doc_id)
                    print(f"  → Создано {count} фрагментов и вычислены эмбеддинги.")
                except Exception as e:
                    print(f"  ⚠ Предупреждение при индексации (возможно, не задан OPENAI_API_KEY): {e}")

def run_evaluation():
    print("=" * 70)
    print(" 🌸 ОЦЕНКА И КАЛИБРОВКА RAG-СИСТЕМЫ DAVINCI FLOWERS 🌸")
    print("=" * 70)

    # 1. Проверяем API ключ
    api_key = rag.get_api_key()
    if not api_key:
        print("\n⚠ ВНИМАНИЕ: OPENAI_API_KEY не задан в .env файле.")
        print("Для вычисления реальных эмбеддингов и генерации ответов укажите ключ в .env.")
        print("Проверка переходит в тестовый режим проверки структуры и базы данных.\n")

    # 2. Подготавливаем тестовые документы
    setup_sample_docs()

    # 3. Читаем вопросы
    q_file = os.path.join(os.path.dirname(__file__), "questions.json")
    with open(q_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    threshold = rag.get_similarity_threshold()
    print(f"\nТекущий установленный порог сходства (threshold): {threshold}\n")

    passed_count = 0
    total_count = len(questions)

    for item in questions:
        q_id = item["id"]
        q_text = item["question"]
        is_trap = item.get("trap", False)

        print("-" * 70)
        print(f"Вопрос #{q_id}: {q_text}")
        print(f"Тип вопроса: {'🚨 ЛОВУШКА (вне темы)' if is_trap else '🌿 Обычный вопрос'}")

        try:
            result = rag.answer_query(q_text, custom_threshold=threshold)
            sim = result.get("similarity", 0.0)
            sources = result.get("sources", [])
            answer = result.get("answer", "")
            mode = result.get("mode", "")

            print(f"Максимальное сходство: {sim:.4f} (порог: {threshold})")
            print(f"Режим ответа: {mode}")
            print(f"Источники: {sources if sources else '—'}")
            print(f"Ответ бота:\n{answer}\n")

            if is_trap:
                # Для ловушки: сходство должно быть ниже порога, модель не должна вызываться
                if sim < threshold or mode == "below_threshold":
                    print("✅ РЕЗУЛЬТАТ: Ловушка успешно отсечена! Бот не вызвал модель и ответил вежливо.")
                    passed_count += 1
                else:
                    print(f"❌ РЕЗУЛЬТАТ: Ловушка не отсечена! Сходство {sim} оказалось выше порога {threshold}.")
            else:
                # Для обычного вопроса: сходство выше порога, источники указаны
                if sim >= threshold or mode in ["rag", "base_florist"]:
                    print("✅ РЕЗУЛЬТАТ: Вопрос успешно обработан!")
                    passed_count += 1
                else:
                    print(f"❌ РЕЗУЛЬТАТ: Вопрос не прошёл порог! Сходство {sim} < {threshold}.")

        except Exception as e:
            print(f"❌ Ошибка при обработке вопроса: {e}")

    print("\n" + "=" * 70)
    print(f"ИТОГ ТЕСТИРОВАНИЯ: Успешно {passed_count} из {total_count} вопросов.")
    print("=" * 70)

if __name__ == "__main__":
    run_evaluation()
