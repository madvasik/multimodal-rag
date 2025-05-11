import os

import streamlit as st
import torch
from dotenv import load_dotenv

from src.llm import chat
from src.retrievers import RetrievePipeline

# --- ШАГ 1: Установить широкий макет ---
# Должно быть первой командой Streamlit
st.set_page_config(layout="wide")
# ---------------------------------------

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

load_dotenv(override=True)


@st.cache_resource
def init_retrieve_pipeline(device: str) -> RetrievePipeline:
    return RetrievePipeline(device=device)


retrieve_pipe = init_retrieve_pipeline(device)


def initialize_session_states() -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []


def sidebar_strategy_selector() -> str:
    st.sidebar.header("Выбор стратегии")
    # Список доступных стратегий
    strategies = ["SummaryEmb", "ColQwen", "ColQwen+SummaryEmb"]
    # Находим индекс стратегии по умолчанию
    default_index = strategies.index("ColQwen+SummaryEmb") if "ColQwen+SummaryEmb" in strategies else 0
    # Возвращаем selectbox с установленным индексом по умолчанию
    return st.sidebar.selectbox(
        "Выберите стратегию поиска:",
        strategies,
        index=default_index # Устанавливаем "ColQwen+SummaryEmb" по умолчанию
    )


def display_chat_history() -> None:
    for message in st.session_state["messages"]:
        role = message["role"]
        content = message["content"]
        with st.chat_message(role):
            if role == "user":
                st.markdown(content)
            else:
                answer_text, image_paths = content
                if image_paths: # Проверяем, есть ли изображения
                    st.markdown("Релевантные изображения по запросу:")
                    # --- Изменения для отображения истории (1 ряд) ---
                    num_images = len(image_paths)
                    # Создаем столько колонок, сколько есть изображений
                    cols = st.columns(num_images)
                    # Распределяем каждое изображение по своей колонке
                    for i, path in enumerate(image_paths):
                        with cols[i]:
                            st.image(
                                path,
                                caption=f"Изображение {i+1}", # Простая нумерация
                                use_container_width=True,
                            )
                    # --- Конец изменений ---
                st.markdown("**Ответ:**\n" + answer_text.lstrip())


def handle_user_query(query: str, strategy: str) -> None:
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
         st.markdown(query)

    with st.status("Обработка запроса...", expanded=False) as status:
        status.update(label="Этап 1/2: Поиск релевантных изображений...")
        image_paths = retrieve_pipe.retrieve(query, strategy)
        # st.write(f"Найдено изображений: {len(image_paths)}")

        status.update(label="Этап 2/2: Генерация ответа по найденным изображениям...")
        structured_query = [{"role": "user", "content": [{"type": "text", "text": query}]}]
        # Передаем image_paths в chat, если они есть, иначе None
        answer_text = chat(structured_query, image_paths if image_paths else None)

        status.update(label="Обработка завершена!", state="complete", expanded=False)

    st.session_state["messages"].append(
        {"role": "assistant", "content": (answer_text, image_paths)}
    )
    with st.chat_message("assistant"):
        if image_paths: # Проверяем, есть ли изображения
            st.markdown("Релевантные изображения по запросу:")
            # --- Изменения для отображения нового ответа (1 ряд) ---
            num_images = len(image_paths)
            # Создаем столько колонок, сколько есть изображений
            cols = st.columns(num_images)
             # Распределяем каждое изображение по своей колонке
            for i, path in enumerate(image_paths):
                 with cols[i]:
                    st.image(
                        path,
                        caption=f"Изображение {i+1}", # Простая нумерация
                        use_container_width=True,
                    )
            # --- Конец изменений ---
        st.markdown("**Ответ:**\n" + answer_text.lstrip())


def main():
    # --- Устанавливаем новый заголовок ---
    st.title("Мультимодальная RAG система 🤖")
    # -----------------------------------

    initialize_session_states()

    # --- Убираем разделитель ---
    # st.sidebar.divider()
    # -------------------------
    clear_chat_button = st.sidebar.button("Очистить историю чата")
    if clear_chat_button:
        st.session_state.messages = []
        st.rerun()

    strategy = sidebar_strategy_selector()

    display_chat_history()

    user_query = st.chat_input("Введите запрос для мультимодального поиска")

    if user_query:
        handle_user_query(user_query, strategy)
        st.rerun()


if __name__ == "__main__":
    main()
