# import streamlit as st
# import logging
# import os
# import shutil
# import pdfplumber
# import ollama
# import warnings
# from langchain_community.document_loaders import UnstructuredPDFLoader
# from langchain_ollama import OllamaEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain.prompts import ChatPromptTemplate, PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_ollama import ChatOllama
# from langchain_core.runnables import RunnablePassthrough
# from langchain.retrievers.multi_query import MultiQueryRetriever
# from typing import List, Tuple, Optional

# # Suppress torch warnings
# warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.classes.*')

# # Set protobuf environment variable to avoid error messages
# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# # Define persistent directory for ChromaDB
# PERSIST_DIRECTORY = os.path.join("data", "vectors")
# DATASET_FOLDER = "dataset"

# # Streamlit page configuration
# st.set_page_config(
#     page_title="Ollama PDF RAG Streamlit UI",
#     page_icon="🎈",
#     layout="wide",
#     initial_sidebar_state="collapsed",
# )

# # Logging configuration
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )
# logger = logging.getLogger(__name__)

# def load_pdfs_from_folder(folder_path: str) -> List[str]:
#     """
#     Load all PDF files from a specified folder.
    
#     Args:
#         folder_path (str): Path to the folder containing PDF files.
    
#     Returns:
#         List[str]: List of PDF file paths.
#     """
#     pdf_files = []
#     if os.path.exists(folder_path):
#         pdf_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.pdf')]
#     return pdf_files


# def create_vector_db_from_folder(pdf_paths: List[str]) -> Chroma:
#     """
#     Create a vector database from multiple PDF files.
    
#     Args:
#         pdf_paths (List[str]): List of PDF file paths.
    
#     Returns:
#         Chroma: A vector store containing the processed document chunks.
#     """
#     logger.info(f"Creating vector DB from multiple files: {len(pdf_paths)} PDFs")
#     all_chunks = []

#     for pdf_path in pdf_paths:
#         loader = UnstructuredPDFLoader(pdf_path)
#         data = loader.load()
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
#         chunks = text_splitter.split_documents(data)
#         all_chunks.extend(chunks)
    
#     embeddings = OllamaEmbeddings(model="nomic-embed-text")
#     vector_db = Chroma.from_documents(
#         documents=all_chunks,
#         embedding=embeddings,
#         persist_directory=PERSIST_DIRECTORY,
#         collection_name="multi_pdf_dataset"
#     )
#     logger.info("Vector DB created with persistent storage")
#     return vector_db


# def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:
#     """
#     Process a user question using the vector database and selected language model.

#     Args:
#         question (str): The user's question.
#         vector_db (Chroma): The vector database containing document embeddings.
#         selected_model (str): The name of the selected language model.

#     Returns:
#         str: The generated response to the user's question.
#     """
#     logger.info(f"Processing question: {question} using model: {selected_model}")
    
#     llm = ChatOllama(model=selected_model)
    
#     QUERY_PROMPT = PromptTemplate(
#         input_variables=["question"],
#         template="""Generate 2 different versions of the given user question
#         to retrieve relevant documents from a vector database.
#         Original question: {question}""",
#     )

#     retriever = MultiQueryRetriever.from_llm(
#         vector_db.as_retriever(),
#         llm,
#         prompt=QUERY_PROMPT
#     )

#     template = """Answer the question based ONLY on the following context:
#     {context}
#     Question: {question}
#     """
#     prompt = ChatPromptTemplate.from_template(template)
    
#     chain = (
#         {"context": retriever, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     response = chain.invoke(question)
#     logger.info("Question processed and response generated")
#     return response


# def main() -> None:
#     """
#     Main function to run the Streamlit application.
#     """
#     st.subheader("🧠PSG Chatbot", divider="gray", anchor=False)

#     models_info = ollama.list()
#     available_models = tuple(model.model for model in models_info.models) if hasattr(models_info, "models") else tuple()

#     if available_models:
#         selected_model = st.selectbox(
#             "Pick a model available locally on your system ↓",
#             available_models,
#             key="model_select"
#         )

#     if "messages" not in st.session_state:
#         st.session_state["messages"] = []
#     if "vector_db" not in st.session_state:
#         pdf_paths = load_pdfs_from_folder(DATASET_FOLDER)
#         if pdf_paths:
#             with st.spinner("Processing multiple PDFs from dataset..."):
#                 st.session_state["vector_db"] = create_vector_db_from_folder(pdf_paths)

#     if "vector_db" not in st.session_state or st.session_state["vector_db"] is None:
#         st.error("No PDF files found in the dataset folder or vector DB not initialized.")
#         return

#     message_container = st.container(height=500, border=True)

#     for i, message in enumerate(st.session_state["messages"]):
#         avatar = "🤖" if message["role"] == "assistant" else "💁🏻"
#         with message_container.chat_message(message["role"], avatar=avatar):
#             st.markdown(message["content"])

#     if prompt := st.chat_input("Enter a prompt here..."):
#         try:
#             st.session_state["messages"].append({"role": "user", "content": prompt})
#             with message_container.chat_message("user", avatar="💁🏻"):
#                 st.markdown(prompt)

#             with message_container.chat_message("assistant", avatar="🤖"):
#                 with st.spinner(":green[processing...]"):
#                     response = process_question(prompt, st.session_state["vector_db"], selected_model)
#                     st.markdown(response)
#                     st.session_state["messages"].append({"role": "assistant", "content": response})

#         except Exception as e:
#             st.error(e, icon="⛔️")
#             logger.error(f"Error processing prompt: {e}")


# if __name__ == "__main__":
#     main()

# # python -m venv venv
# # .\venv\Scripts\activate
# # pip install -r requirements.txt
import streamlit as st
import logging
import os
import io
import warnings
import speech_recognition as sr
from st_audiorec import st_audiorec
from typing import List

import ollama
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.classes.*')
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

PERSIST_DIRECTORY = os.path.join("data", "vectors")
DATASET_FOLDER = "dataset"

st.set_page_config(
    page_title="Ollama PDF RAG Streamlit UI",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_pdfs_from_folder(folder_path: str) -> List[str]:
    if os.path.exists(folder_path):
        return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.pdf')]
    return []

def create_vector_db_from_folder(pdf_paths: List[str]) -> Chroma:
    logger.info(f"Creating vector DB from {len(pdf_paths)} PDFs")
    all_chunks = []
    for pdf_path in pdf_paths:
        loader = UnstructuredPDFLoader(pdf_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(data)
        all_chunks.extend(chunks)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name="multi_pdf_dataset"
    )

def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:
    logger.info(f"Processing question: {question}")
    llm = ChatOllama(model=selected_model)

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="Generate 2 different versions of the given user question to retrieve relevant documents.\nOriginal question: {question}",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )

    prompt_template = ChatPromptTemplate.from_template(
        "Answer the question based ONLY on the following context:\n{context}\nQuestion: {question}"
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return chain.invoke(question)

def transcribe_audio(audio_data) -> str:
    audio_file = io.BytesIO(audio_data)
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Speech recognition error: {e}"

def main():
    st.subheader("🧠 PSG Chatbot", divider="gray")

    models_info = ollama.list()
    available_models = tuple(model.model for model in models_info.models) if hasattr(models_info, "models") else ()
    selected_model = st.selectbox("Pick a local model ↓", available_models)

    if "vector_db" not in st.session_state:
        pdf_paths = load_pdfs_from_folder(DATASET_FOLDER)
        if pdf_paths:
            with st.spinner("Loading PDFs..."):
                st.session_state["vector_db"] = create_vector_db_from_folder(pdf_paths)

    if "vector_db" not in st.session_state or st.session_state["vector_db"] is None:
        st.error("No PDF files found in the dataset folder or vector DB not initialized.")
        return

    if "recording" not in st.session_state:
        st.session_state["recording"] = False
    if "voice_input_text" not in st.session_state:
        st.session_state["voice_input_text"] = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    col1, col2 = st.columns([0.9, 0.1])
    with col1:
        prompt_input = st.text_input("Search your question...", key="searchbox")
    with col2:
        if st.button("🎙️" if not st.session_state["recording"] else "⏹️", use_container_width=True):
            st.session_state["recording"] = not st.session_state["recording"]

    if st.session_state["recording"]:
        st.markdown("#### Speak Now...")
        wav_audio_data = st_audiorec()
        if wav_audio_data is not None:
            st.audio(wav_audio_data, format="audio/wav")
            transcription = transcribe_audio(wav_audio_data)
            st.session_state["voice_input_text"] = transcription
            st.session_state["recording"] = False
            st.rerun()

    # Use transcribed voice input if available
    if st.session_state.voice_input_text:
        final_prompt = st.session_state.voice_input_text
        st.session_state.voice_input_text = ""
    else:
        final_prompt = prompt_input

    # Show chat interface
    message_container = st.container(height=500, border=True)

    if final_prompt:
        with message_container.chat_message("user", avatar="💁🏻"):
            st.markdown(final_prompt)
        try:
            with message_container.chat_message("assistant", avatar="🤖"):
                with st.spinner("Processing..."):
                    response = process_question(final_prompt, st.session_state.vector_db, selected_model)
                    st.markdown(response)
            st.session_state.chat_history.append((final_prompt, response))
        except Exception as e:
            st.error(str(e), icon="⛔️")
            logger.exception("Error in processing")

    # Display chat history cleanly
    for user_msg, bot_msg in st.session_state.chat_history[:-1]:
        with message_container.chat_message("user", avatar="💁🏻"):
            st.markdown(user_msg)
        with message_container.chat_message("assistant", avatar="🤖"):
            st.markdown(bot_msg)

if __name__ == "__main__":
    main()
