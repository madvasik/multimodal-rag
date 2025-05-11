import os
import glob
import json
import faiss
import torch
import unicodedata
from transformers import AutoModel, AutoTokenizer
from omegaconf import OmegaConf
from dotenv import load_dotenv
import sys
import numpy as np # Добавлено для работы с эмбеддингами

# Добавляем корневую директорию проекта в PYTHONPATH, чтобы можно было импортировать src.llm
# Это может потребовать корректировки в зависимости от того, откуда вы запускаете скрипт
script_dir = os.path.dirname(os.path.abspath(__file__))
# Переходим на два уровня вверх от директории скрипта (retrievers -> src -> корень проекта)
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)

try:
    from src.llm import summarize_image
except ImportError:
    print("Ошибка: Не удалось импортировать summarize_image из src.llm.")
    print("Убедитесь, что скрипт находится в папке scripts, и структура проекта верна.")
    print(f"Project root added to sys.path: {project_root}")
    sys.exit(1)


# --- Конфигурация ---
load_dotenv(override=True)
# Можно оставить или убрать эту строку после проверки
# print(f"DEBUG: BGE_CONFIG_PATH после load_dotenv: {os.getenv('BGE_CONFIG_PATH')}")

# Убедитесь, что переменная BGE_CONFIG_PATH установлена в вашем .env файле
# и указывает на ОБНОВЛЕННЫЙ конфиг (например, data/index_BGE_new/)
BGE_CONFIG_PATH = os.getenv("BGE_CONFIG_PATH")
if not BGE_CONFIG_PATH:
    print("Ошибка: Переменная окружения BGE_CONFIG_PATH не установлена.")
    print("Добавьте BGE_CONFIG_PATH=путь/к/вашему/bge_config.yaml в .env файл.")
    sys.exit(1)
if not os.path.exists(BGE_CONFIG_PATH):
    print(f"Ошибка: Файл конфигурации BGE не найден по пути: {BGE_CONFIG_PATH}")
    sys.exit(1)

# PDF_NAME больше не нужен, скрипт будет обрабатывать все подпапки в images_path
# PDF_NAME = "bmw_i8" # Имя подпапки с изображениями PDF
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Можно принудительно установить CPU: DEVICE = "cpu"

# --- Основная логика ---
def build_index():
    print(f"Используется устройство: {DEVICE}")
    print(f"Загрузка конфигурации BGE из: {BGE_CONFIG_PATH}")

    try:
        bge_config = OmegaConf.load(BGE_CONFIG_PATH)
    except Exception as e:
        print(f"Ошибка загрузки конфигурации BGE: {e}")
        sys.exit(1)

    # Проверка наличия необходимых ключей в конфигурации
    required_keys = ["model_name", "images_path", "faiss_path", "metadata_path"]
    if not all(key in bge_config for key in required_keys):
        print(f"Ошибка: В файле конфигурации {BGE_CONFIG_PATH} отсутствуют необходимые ключи.")
        print(f"Требуемые ключи: {required_keys}")
        sys.exit(1)

    # --- Начало изменений ---
    base_images_path = bge_config.images_path
    print(f"Базовая директория с изображениями: {base_images_path}")
    if not os.path.isdir(base_images_path):
        print(f"Ошибка: Базовая директория с изображениями не найдена: {base_images_path}")
        sys.exit(1)

    # Находим все поддиректории (считаем их именами PDF)
    try:
        pdf_names = [d for d in os.listdir(base_images_path) if os.path.isdir(os.path.join(base_images_path, d))]
    except OSError as e:
        print(f"Ошибка чтения поддиректорий в {base_images_path}: {e}")
        sys.exit(1)

    if not pdf_names:
        print(f"Предупреждение: Не найдено поддиректорий (документов) в {base_images_path}")
        # Можно либо завершить скрипт, либо создать пустые файлы
        # sys.exit(0)

    print(f"Найдены следующие директории документов для обработки: {pdf_names}")
    # --- Конец изменений ---

    # Убеждаемся, что директории для сохранения индекса и метаданных существуют
    faiss_dir = os.path.dirname(bge_config.faiss_path)
    meta_dir = os.path.dirname(bge_config.metadata_path)
    try:
        os.makedirs(faiss_dir, exist_ok=True)
        os.makedirs(meta_dir, exist_ok=True)
    except OSError as e:
        print(f"Ошибка создания директорий для вывода: {e}")
        sys.exit(1)

    print(f"Целевой FAISS индекс: {bge_config.faiss_path}")
    print(f"Целевые метаданные: {bge_config.metadata_path}")

    # Загружаем модель BGE и токенизатор
    print(f"Загрузка модели BGE: {bge_config.model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(bge_config.model_name)
        model = AutoModel.from_pretrained(bge_config.model_name).to(DEVICE)
        model.eval() # Переводим модель в режим оценки
    except Exception as e:
        print(f"Ошибка загрузки модели или токенизатора BGE: {e}")
        sys.exit(1)

    # Определяем размерность эмбеддинга
    embedding_dim = model.config.hidden_size
    print(f"Инициализация FAISS индекса (размерность={embedding_dim})")
    index = faiss.IndexFlatL2(embedding_dim) # Используем L2 расстояние
    metadata = []
    all_embeddings = [] # Список для сбора всех эмбеддингов
    total_images_processed = 0
    total_images_skipped = 0

    # --- Начало изменений ---
    # Итерируемся по каждой найденной папке (документу)
    for pdf_name in pdf_names:
        source_image_dir = os.path.join(base_images_path, pdf_name)
        print(f"\nОбработка директории: {source_image_dir}")

        # Находим и сортируем файлы изображений для текущего документа
        image_files = sorted(glob.glob(os.path.join(source_image_dir, "*.png")))
        num_images = len(image_files)

        if num_images == 0:
            print(f"  Предупреждение: Не найдено .png файлов в {source_image_dir}")
            continue

        print(f"  Найдено {num_images} изображений для обработки.")
        current_doc_images_processed = 0

        # Обрабатываем каждое изображение в текущем документе
        for i, image_path in enumerate(image_files):
            base_name = os.path.basename(image_path)
            print(f"  Обработка изображения {i+1}/{num_images}: {base_name}...")
            try:
                # 1. Суммаризация изображения
                print("    Генерация summary...")
                summary = summarize_image(image_path)
                if not summary:
                    print(f"    Предупреждение: Не удалось сгенерировать summary для {image_path}. Пропуск.")
                    total_images_skipped += 1
                    continue
                # print(f"    Summary: {summary[:100]}...") # Для отладки

                # 2. Создание эмбеддинга для summary
                print("    Генерация эмбеддинга...")
                inputs = tokenizer(
                    [summary], return_tensors="pt", padding=True, truncation=True, max_length=512 # Ограничим длину для BGE
                ).to(DEVICE)
                with torch.no_grad():
                    outputs = model(**inputs)

                embedding = outputs.last_hidden_state[:, 0, :]
                # Собираем эмбеддинги в список
                all_embeddings.append(embedding.float().cpu().numpy())

                # 4. Добавление метаданных (добавляем сразу, т.к. порядок важен)
                jpeg_name = base_name
                pdf_name_normalized = unicodedata.normalize("NFC", pdf_name) # Используем pdf_name из цикла
                metadata.append({"pdf": pdf_name_normalized, "jpeg": jpeg_name})
                print(f"    Эмбеддинг и метаданные подготовлены.")
                current_doc_images_processed += 1

            except Exception as e:
                print(f"    Ошибка при обработке {image_path}: {e}")
                total_images_skipped += 1
                # Можно добавить более детальное логирование ошибок

        total_images_processed += current_doc_images_processed
        print(f"  Обработано {current_doc_images_processed} изображений в этой директории.")
    # --- Конец изменений ---

    print(f"\nВсего обработано изображений: {total_images_processed}")
    print(f"Всего пропущено изображений (из-за ошибок или отсутствия summary): {total_images_skipped}")

    # Сохраняем индекс и метаданные, если есть что сохранять
    if all_embeddings:
        print(f"\nОбъединение {len(all_embeddings)} эмбеддингов...")
        embeddings_matrix = np.vstack(all_embeddings) # Объединяем список эмбеддингов в одну матрицу
        print(f"Размер итоговой матрицы эмбеддингов: {embeddings_matrix.shape}")

        print(f"Добавление эмбеддингов в FAISS индекс...")
        index.add(embeddings_matrix)
        print(f"Итоговый размер FAISS индекса (ntotal): {index.ntotal}")

        print(f"Сохранение FAISS индекса в {bge_config.faiss_path}")
        try:
            faiss.write_index(index, bge_config.faiss_path)
        except Exception as e:
            print(f"Ошибка сохранения FAISS индекса: {e}")

        # Проверяем соответствие количества метаданных и эмбеддингов в индексе
        if len(metadata) == index.ntotal:
            print(f"Сохранение метаданных ({len(metadata)} записей) в {bge_config.metadata_path}")
            try:
                with open(bge_config.metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=4)
            except Exception as e:
                 print(f"Ошибка сохранения файла метаданных: {e}")
        else:
            print(f"КРИТИЧЕСКОЕ ПРЕДУПРЕЖДЕНИЕ: Количество метаданных ({len(metadata)}) не совпадает с количеством эмбеддингов в FAISS индексе ({index.ntotal}). Файл метаданных НЕ сохранен из-за несоответствия!")

    else:
        print("\nЭмбеддинги не были сгенерированы или собраны. Файлы индекса и метаданных не сохранены.")

    print("\nСкрипт завершен.")

if __name__ == "__main__":
    build_index()
