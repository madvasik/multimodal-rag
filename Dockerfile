FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY streamlit_ui ./streamlit_ui
COPY scripts ./scripts

RUN pip install --no-cache-dir -U pip setuptools wheel \
    && pip install --no-cache-dir -e .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_ui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
