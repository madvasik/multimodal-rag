FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY streamlit_ui ./streamlit_ui
COPY scripts ./scripts

# Official PyTorch CPU wheels (tag pytorch/pytorch:2.3.1-cpu is not published on Docker Hub).
RUN pip install --no-cache-dir -U pip setuptools wheel \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_ui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
