import os
import glob
import json
import unicodedata
from omegaconf import OmegaConf
from dotenv import load_dotenv
import sys
import torch # Добавлено для проверки эмбеддингов

# --- Конфигурация ---
load_dotenv(override=True) # Загружаем переменные окружения из .env файла, перезаписывая существующие

# Убедитесь, что переменная COLQWEN_CONFIG_PATH установлена в вашем .env файле
# и указывает на ОБНОВЛЕННЫЙ конфиг (например, data/index_colqwen_new/)
COLQWEN_CONFIG_PATH = os.getenv("COLQWEN_CONFIG_PATH")
if not COLQWEN_CONFIG_PATH:
    print("Ошибка: Переменная окружения COLQWEN_CONFIG_PATH не установлена.")
    print("Добавьте COLQWEN_CONFIG_PATH=путь/к/вашему/colqwen_config.yaml в .env файл.")
    sys.exit(1)
if not os.path.exists(COLQWEN_CONFIG_PATH):
    print(f"Ошибка: Файл конфигурации ColQwen не найден по пути: {COLQWEN_CONFIG_PATH}")
    sys.exit(1)

# PDF_NAME больше не нужен, скрипт будет обрабатывать все подпапки в images_path
# PDF_NAME = "bmw_i8" # Имя подпапки с изображениями PDF

# --- Основная логика ---
def build_metadata():
    print(f"Загрузка конфигурации ColQwen из: {COLQWEN_CONFIG_PATH}")

    try:
        colqwen_config = OmegaConf.load(COLQWEN_CONFIG_PATH)
    except Exception as e:
        print(f"Ошибка загрузки конфигурации ColQwen: {e}")
        sys.exit(1)

    # Проверка наличия необходимых ключей в конфигурации
    required_keys = ["images_path", "metadata_path", "embeddings_path"] # Убедимся, что все пути есть
    if not all(key in colqwen_config for key in required_keys):
        print(f"Ошибка: В файле конфигурации {COLQWEN_CONFIG_PATH} отсутствуют необходимые ключи.")
        print(f"Требуемые ключи: {required_keys}")
        sys.exit(1)

    # --- Начало изменений ---
    base_images_path = colqwen_config.images_path
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
        # Создаем пустой файл метаданных или выходим
        # sys.exit(0) # или можно просто продолжить и создать пустой json

    print(f"Найдены следующие директории документов: {pdf_names}")
    # --- Конец изменений ---


    # Убеждаемся, что директория для сохранения метаданных существует
    meta_dir = os.path.dirname(colqwen_config.metadata_path)
    try:
        os.makedirs(meta_dir, exist_ok=True)
    except OSError as e:
        print(f"Ошибка создания директории для метаданных: {e}")
        sys.exit(1)

    print(f"Целевой файл метаданных: {colqwen_config.metadata_path}")

    metadata = []
    total_images_processed = 0

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
            continue # Пропускаем эту папку, если в ней нет изображений

        print(f"  Найдено {num_images} изображений.")
        total_images_processed += num_images

        # Создаем метаданные для каждого изображения в этой папке
        for image_path in image_files:
            base_name = os.path.basename(image_path)
            jpeg_name = base_name # Используем имя файла как есть
            # Используем имя подпапки как имя PDF
            pdf_name_normalized = unicodedata.normalize("NFC", pdf_name)
            metadata.append({"pdf": pdf_name_normalized, "jpeg": jpeg_name})
    # --- Конец изменений ---


    # Сохраняем собранные метаданные
    print(f"\nСохранение метаданных ({len(metadata)} записей) в {colqwen_config.metadata_path}")
    if not metadata:
        print("Предупреждение: Нет данных для сохранения в файл метаданных.")
    try:
        with open(colqwen_config.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4) # Добавляем отступы для читаемости
    except Exception as e:
         print(f"Ошибка сохранения файла метаданных: {e}")

    # Проверка соответствия количества метаданных и эмбеддингов
    # ВАЖНО: Эта проверка предполагает, что все эмбеддинги лежат в ОДНОЙ папке embeddings_path
    # и их порядок СООТВЕТСТВУЕТ порядку обхода папок документов и файлов изображений в них (с сортировкой!)
    try:
        embedding_files = sorted(glob.glob(os.path.join(colqwen_config.embeddings_path, "*.pt")))
        total_embeddings_in_files = 0 # Переменная для подсчета общего числа эмбеддингов в файлах

        if embedding_files:
            print(f"Найдено {len(embedding_files)} файлов с эмбеддингами ColQwen в {colqwen_config.embeddings_path}.")
            # Пытаемся посчитать общее количество эмбеддингов, если файлы не слишком большие
            # ОСТОРОЖНО: Загрузка всех эмбеддингов может требовать много памяти!
            # Раскомментируйте и адаптируйте, если нужно точное сравнение.
            # for emb_file in embedding_files:
            #     try:
            #         embeddings_chunk = torch.load(emb_file, map_location='cpu') # Загружаем на CPU для экономии VRAM
            #         total_embeddings_in_files += embeddings_chunk.shape[0]
            #     except Exception as load_err:
            #         print(f"  Ошибка загрузки файла эмбеддингов {emb_file}: {load_err}")

            # Пока просто сравним количество записей метаданных с количеством обработанных изображений
            print(f"Общее количество обработанных изображений: {total_images_processed}")
            # print(f"Общее количество найденных эмбеддингов (если удалось загрузить): {total_embeddings_in_files}") # Раскомментировать, если считали

            if len(metadata) != total_images_processed:
                 print(f"ПРЕДУПРЕЖДЕНИЕ: Количество записей в метаданных ({len(metadata)}) не совпадает с количеством обработанных PNG файлов ({total_images_processed}).")
            # Дополнительная проверка, если считали эмбеддинги:
            # if total_embeddings_in_files > 0 and len(metadata) != total_embeddings_in_files:
            #      print(f"КРИТИЧЕСКОЕ ПРЕДУПРЕЖДЕНИЕ: Количество записей в метаданных ({len(metadata)}) не совпадает с общим количеством эмбеддингов в файлах ({total_embeddings_in_files}). Проверьте процесс генерации/сортировки эмбеддингов и изображений!")

        else:
             print(f"Предупреждение: Файлы эмбеддингов ColQwen (*.pt) не найдены в {colqwen_config.embeddings_path}. Сравнение невозможно.")


    except Exception as e:
        print(f"Не удалось проверить количество эмбеддингов: {e}")


    print("\nСкрипт завершен.")

if __name__ == "__main__":
    build_metadata()
