import os

from dotenv import load_dotenv

load_dotenv(override=True)

import streamlit as st

from src.mistral_api import chat
from src.retrieval import RetrievePipeline

st.set_page_config(layout="wide")

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

_INIT_HINT = (
    "Нужны: PNG в `data/images/<документ>/`, файл `data/index_text/faiss_index.bin`, "
    "шарды `data/index_visual/embeddings/embeddings_*.pt` и согласованный `docs_meta.json`. "
    "Сборка: `scripts/build_indexes/` и раздел «Пайплайн индексов» в README."
)


@st.cache_resource
def load_retrieve_pipeline() -> RetrievePipeline:
    return RetrievePipeline(device="cpu")


def initialize_session_states() -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []


def sidebar_strategy_selector() -> str:
    st.sidebar.header("Выбор стратегии")
    strategies = ["SummaryEmb", "ColQwen", "ColQwen+SummaryEmb"]
    default_index = strategies.index("ColQwen+SummaryEmb") if "ColQwen+SummaryEmb" in strategies else 0
    return st.sidebar.selectbox("Выберите стратегию поиска:", strategies, index=default_index)


def display_chat_history() -> None:
    for message in st.session_state["messages"]:
        role = message["role"]
        content = message["content"]
        with st.chat_message(role):
            if role == "user":
                st.markdown(content)
            else:
                answer_text, image_paths = content
                if image_paths:
                    st.markdown("Релевантные изображения по запросу:")
                    num_images = len(image_paths)
                    cols = st.columns(num_images)
                    for i, path in enumerate(image_paths):
                        with cols[i]:
                            st.image(path, caption=f"Изображение {i+1}", use_container_width=True)
                st.markdown("**Ответ:**\n" + answer_text.lstrip())


def handle_user_query(query: str, strategy: str, retrieve_pipe: RetrievePipeline) -> None:
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.status("Обработка запроса...", expanded=False) as status:
        status.update(label="Этап 1/2: Поиск релевантных изображений...")
        image_paths = retrieve_pipe.retrieve(query, strategy)
        status.update(label="Этап 2/2: Генерация ответа по найденным изображениям...")
        structured_query = [{"role": "user", "content": [{"type": "text", "text": query}]}]
        answer_text = chat(structured_query, image_paths if image_paths else None)
        status.update(label="Обработка завершена!", state="complete", expanded=False)

    st.session_state["messages"].append({"role": "assistant", "content": (answer_text, image_paths)})
    with st.chat_message("assistant"):
        if image_paths:
            st.markdown("Релевантные изображения по запросу:")
            num_images = len(image_paths)
            cols = st.columns(num_images)
            for i, path in enumerate(image_paths):
                with cols[i]:
                    st.image(path, caption=f"Изображение {i+1}", use_container_width=True)
        st.markdown("**Ответ:**\n" + answer_text.lstrip())


def main():
    st.title("Мультимодальная RAG система 🤖")
    initialize_session_states()
    clear_chat_button = st.sidebar.button("Очистить историю чата")
    if clear_chat_button:
        st.session_state.messages = []
        st.rerun()

    try:
        retrieve_pipe = load_retrieve_pipeline()
    except Exception as e:
        st.error(f"Не удалось загрузить пайплайн поиска:\n\n`{e}`\n\n{_INIT_HINT}")
        st.stop()

    strategy = sidebar_strategy_selector()
    display_chat_history()
    user_query = st.chat_input("Введите запрос для мультимодального поиска")

    if user_query:
        handle_user_query(user_query, strategy, retrieve_pipe)
        st.rerun()


if __name__ == "__main__":
    main()
