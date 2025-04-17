import streamlit as st
import logging
import os
import io
import warnings
import speech_recognition as sr
from st_audiorec import st_audiorec
from typing import List
from datetime import datetime
from fpdf import FPDF
import hashlib
from PyPDF2 import PdfReader, PdfWriter
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
LIKED_PDF_PATH = os.path.join(DATASET_FOLDER, "liked_qa.pdf")

st.set_page_config(
    page_title="PSG Llama Chat: AI At Your Service",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def hash_file(path):
    hasher = hashlib.md5()
    with open(path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def load_pdfs_from_folder(folder_path: str) -> List[str]:
    if os.path.exists(folder_path):
        return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.pdf')]
    return []

def create_or_update_vector_db(pdf_paths: List[str]) -> Chroma:
    logger.info(f"Creating/updating vector DB from {len(pdf_paths)} PDFs")

    if "pdf_hashes" not in st.session_state:
        st.session_state["pdf_hashes"] = {}

    new_chunks = []
    for pdf_path in pdf_paths:
        current_hash = hash_file(pdf_path)
        if st.session_state["pdf_hashes"].get(pdf_path) == current_hash:
            continue
        loader = UnstructuredPDFLoader(pdf_path)
        data = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(data)
        new_chunks.extend(chunks)
        st.session_state["pdf_hashes"][pdf_path] = current_hash

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    if os.path.exists(PERSIST_DIRECTORY):
        vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        if new_chunks:
            vectordb.add_documents(new_chunks)
            vectordb.persist()
    else:
        vectordb = Chroma.from_documents(
            documents=new_chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY,
            collection_name="multi_pdf_dataset"
        )

    return vectordb

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

def save_to_pdf(question: str, answer: str):
    if not os.path.exists(DATASET_FOLDER):
        os.makedirs(DATASET_FOLDER)

    temp_pdf = "temp_liked.pdf"
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.multi_cell(0, 10, f"Time: {timestamp}")
    pdf.multi_cell(0, 10, f"Question: {question}")
    pdf.multi_cell(0, 10, f"Answer: {answer}")
    pdf.ln(5)
    pdf.output(temp_pdf)

    writer = PdfWriter()
    if os.path.exists(LIKED_PDF_PATH):
        existing_pdf = PdfReader(LIKED_PDF_PATH)
        for page in existing_pdf.pages:
            writer.add_page(page)

    new_pdf = PdfReader(temp_pdf)
    for page in new_pdf.pages:
        writer.add_page(page)

    with open(LIKED_PDF_PATH, "wb") as f:
        writer.write(f)

    os.remove(temp_pdf)

def main():
    if "vector_db_built" not in st.session_state:
        st.session_state.vector_db_built = False
    if "question_processed" not in st.session_state:
        st.session_state.question_processed = False
    if "recording" not in st.session_state:
        st.session_state["recording"] = False
    if "voice_input_text" not in st.session_state:
        st.session_state["voice_input_text"] = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "final_prompt_ready" not in st.session_state:
        st.session_state["final_prompt_ready"] = False

    st.subheader("🏫 PSG Llama Chat", divider="gray")

    models_info = ollama.list()
    available_models = tuple(model.model for model in models_info.models) if hasattr(models_info, "models") else ()
    selected_model = st.selectbox("Pick a local model ↓", available_models)

    if not st.session_state.vector_db_built:
        pdf_paths = load_pdfs_from_folder(DATASET_FOLDER)
        if pdf_paths:
            with st.spinner("Loading PDFs..."):
                st.session_state.vector_db = create_or_update_vector_db(pdf_paths)
                st.session_state.vector_db_built = True

    col1, col2, col3 = st.columns([0.55, 0.2, 0.25])

    def set_final_prompt():
        st.session_state["final_prompt_ready"] = True
        st.session_state["current_prompt"] = st.session_state["searchbox"]

    with col1:
        st.text_input(
            "Search your question...",
            key="searchbox",
            on_change=set_final_prompt,
        )

    with col2:
        if st.button("Send 📩", use_container_width=True):
            set_final_prompt()

    with col3:
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

    final_prompt = ""
    if st.session_state.final_prompt_ready:
        final_prompt = st.session_state.searchbox.strip()
        st.session_state.final_prompt_ready = False
    elif st.session_state.voice_input_text:
        final_prompt = st.session_state.voice_input_text.strip()
        st.session_state.voice_input_text = ""

    if final_prompt != st.session_state.get("current_prompt", ""):
        st.session_state.question_processed = False

    message_container = st.container(height=500, border=True)

    if final_prompt and not st.session_state.question_processed:
        with message_container.chat_message("user", avatar="🙋🏻‍♂️"):
            st.markdown(final_prompt)

        try:
            with message_container.chat_message("assistant", avatar="🤖"):
                with st.spinner("Processing..."):
                    response = process_question(final_prompt, st.session_state.vector_db, selected_model)
                    st.markdown(response)

                st.session_state.current_prompt = final_prompt
                st.session_state.current_response = response
                st.session_state.chat_history.append((final_prompt, response))
                st.session_state.question_processed = True
        except Exception as e:
            st.error(str(e), icon="⛔")
            logger.exception("Error in processing")

    if "current_prompt" in st.session_state and "current_response" in st.session_state:
        col_like, col_dislike = st.columns([0.2, 0.2])
        with col_like:
            if st.button("👍🏻 Like"):
                save_to_pdf(st.session_state.current_prompt, st.session_state.current_response)
                st.success("Saved to liked_qa.pdf")
        with col_dislike:
            if st.button("👎🏻 Dislike"):
                pass

    for user_msg, bot_msg in st.session_state.chat_history[:-1]:
        with message_container.chat_message("user", avatar="🙋🏻‍♂️"):
            st.markdown(user_msg)
        with message_container.chat_message("assistant", avatar="🤖"):
            st.markdown(bot_msg)

if __name__ == "__main__":
    main()
