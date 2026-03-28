import os
from typing import Dict, List, Optional

from mistralai.client import Mistral

from src.utils import encode_image, image_data_url, load_prompts_from_yaml

_client: Optional[Mistral] = None
_prompts: Optional[dict] = None
_model: Optional[str] = None


def _ensure_mistral_client() -> Mistral:
    global _client
    if _client is None:
        key = os.getenv("MISTRAL_API_KEY")
        if not key:
            raise RuntimeError("MISTRAL_API_KEY is not set.")
        _client = Mistral(api_key=key)
    return _client


def _ensure_prompts_and_model() -> tuple[dict, str]:
    global _prompts, _model
    if _prompts is None:
        path = os.getenv("PROMPTS_PATH")
        if not path:
            raise RuntimeError(
                "PROMPTS_PATH is not set (e.g. src/mistral_api/prompts.yaml or /app/src/mistral_api/prompts.yaml in Docker)."
            )
        _prompts = load_prompts_from_yaml(path)
    if _model is None:
        model = os.getenv("MODEL_NAME")
        if not model:
            raise RuntimeError(
                "MODEL_NAME is not set (e.g. pixtral-12b-2409 for vision + chat)."
            )
        _model = model
    return _prompts, _model


def chat(chat_history: List[Dict], images: List[str] = None):
    prompts, model = _ensure_prompts_and_model()
    client = _ensure_mistral_client()
    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompts["system"]}]}
    ]
    messages += chat_history
    if images:
        for image_path in images:
            base64_image = encode_image(image_path=image_path)
            if not base64_image:
                raise RuntimeError(f"Не удалось прочитать изображение: {image_path}")
            messages[-1]["content"].append(
                {
                    "type": "image_url",
                    "image_url": image_data_url(image_path, base64_image),
                }
            )
    chat_response = client.chat.complete(model=model, messages=messages)
    return chat_response.choices[0].message.content


def summarize_image(image_path: str):
    """
    Отправляет изображение в модель Pixtral-12B для суммаризации.
    """
    prompts, model = _ensure_prompts_and_model()
    client = _ensure_mistral_client()
    base64_image = encode_image(image_path)
    if not base64_image:
        raise RuntimeError(f"Не удалось прочитать изображение: {image_path}")
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": prompts["summary"]}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "summary"},
                {
                    "type": "image_url",
                    "image_url": image_data_url(image_path, base64_image),
                },
            ],
        },
    ]

    chat_response = client.chat.complete(model=model, messages=messages)
    return chat_response.choices[0].message.content
