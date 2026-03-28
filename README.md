# Мультимодальная RAG система

## Обзор

Два типа эмбеддингов для поиска по страницам документов (PNG):

| Тип | Как устроено | Файлы / модули | Стратегия в UI |
|-----|----------------|----------------|----------------|
| **Текстовые** | BGE по саммари страниц (Mistral смотрит картинку → текст → BGE → FAISS) | `build_text_faiss_index.py`, `BGERetriever` | **SummaryEmb** |
| **Визуальные** | ColQwen по пикселям | `build_visual_embeddings.py`, `ColQwenRetriever` | **ColQwen** |

**ColQwen+SummaryEmb** — оба канала. Чат и саммари страниц — **Mistral** (`src/mistral_api/`). Запуск UI: **Docker** или локально `streamlit`.

## Структура репозитория

```text
multimodal-rag/
├── streamlit_ui/
│   └── app.py                  # точка входа Streamlit
├── scripts/
│   ├── prepare_documents/      # подготовка PDF/DOCX, zip
│   │   ├── sort_loose_pdfs_docx.py
│   │   ├── pdf_to_images.py
│   │   └── zip_or_unzip_folder.py
│   ├── build_indexes/          # сборка индексов
│   │   └── build_visual_embeddings.py
│   └── decompress_json_gz.py
├── src/
│   ├── config/
│   │   ├── text_index.yaml     # FAISS + BGE (текстовый канал)
│   │   └── visual_index.yaml   # ColQwen (визуальный канал)
│   ├── mistral_api/            # чат, summarize_image, prompts.yaml
│   └── retrieval/              # пайплайн поиска, FAISS, ColQwen
│       ├── multimodal_search.py
│       ├── build_text_faiss_index.py
│       └── build_visual_metadata.py
├── data/
│   ├── images/                 # PNG по папкам документов
│   ├── index_text/             # FAISS + meta (текстовый индекс)
│   └── index_visual/           # эмбеддинги + meta (визуальный индекс)
├── Dockerfile
├── docker-compose.yml
├── docker-compose.gpu.yml
└── run_app.sh
```

## Переменные окружения

| Переменная | Назначение |
|------------|------------|
| `TEXT_INDEX_CONFIG_PATH` | YAML текстового индекса (BGE + FAISS) |
| `VISUAL_INDEX_CONFIG_PATH` | YAML визуального индекса (ColQwen) |
| `PROMPTS_PATH` | `src/mistral_api/prompts.yaml` |
| `MISTRAL_API_KEY` | ключ API |
| `MODEL_NAME` | vision-модель (напр. `pixtral-12b-2409`) |

Поддерживаются устаревшие имена: `BGE_CONFIG_PATH` → текстовый YAML, `COLQWEN_CONFIG_PATH` → визуальный YAML.

## Быстрый старт (Docker)

```bash
cp .env.example .env
# укажите MISTRAL_API_KEY в .env
./run_app.sh
```

Порт **8501**. В контейнере пути к конфигам задаёт `docker-compose.yml` (`/app/src/config/...`).

### GPU (NVIDIA, Linux)

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

## Локально

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e .
cp .env.example .env
streamlit run streamlit_ui/app.py
```

## Пайплайн индексов

```bash
python scripts/prepare_documents/sort_loose_pdfs_docx.py --work-dir .
python scripts/prepare_documents/pdf_to_images.py --pdf-dir path/to/pdf_files --output-dir data/images

python scripts/build_indexes/build_visual_embeddings.py
python src/retrieval/build_visual_metadata.py

python src/retrieval/build_text_faiss_index.py
```

В Docker: `docker compose run --rm app python <команда>`.

Распаковка `*.json.gz` в каталоге:

```bash
python scripts/decompress_json_gz.py --input-dir path/to/dir --output-dir path/to/out
```

## Проверка

- `python -m compileall src scripts streamlit_ui`
- `python -c "from src.retrieval import RetrievePipeline"`
