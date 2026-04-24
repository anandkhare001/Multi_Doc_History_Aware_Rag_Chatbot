from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import List
from langchain_core.documents import Document
import os
from chroma_utils import get_vectorstore
from dotenv import load_dotenv

# Load environment variables from .env file, especially for the OpenAI API key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# Ollama base URL — overridden by Docker env var when running in containers
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Chat models that should be routed to Ollama instead of OpenAI
OLLAMA_CHAT_MODELS = {"gemma:2b-instruct"}

output_parser = StrOutputParser()


# Setting Up Prompts
# --------------------------------------------------
# Prompt to reformulate user question into a standalone question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Prompt template for question answering using retrieved context
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the following context to answer the user's question."),
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])


# ── LLM Factory ────────────────────────────────────────────────────────────────

def get_llm(model: str):
    """
    Return the appropriate LLM based on the model name.

    Local models (Ollama):
        - gemma:2b-instruct  →  ChatOllama pointing at OLLAMA_BASE_URL

    API models (OpenAI):
        - gpt-3.5-turbo, gpt-4o, gpt-4o-mini  →  ChatOpenAI

    Args:
        model (str): Model identifier string from ModelName enum.

    Returns:
        A LangChain chat model instance.
    """
    if model in OLLAMA_CHAT_MODELS:
        return ChatOllama(
            model=model,
            base_url=OLLAMA_BASE_URL,
            temperature=0.2           # lower temp for more factual RAG answers
        )
    return ChatOpenAI(model=model)


# Creating the Retrieval-Augmented Generation Chain
# --------------------------------------------------
def get_rag_chain(model: str = "gpt-3.5-turbo",
                  embedding_model: str = "openai"):
    """
    Build and return a history-aware RAG chain.

    The retriever is initialised from the vectorstore that matches the
    embedding_model so queries are always embedded with the same model
    that was used during document indexing.

    Args:
        model (str):           Chat model identifier (ModelName enum value).
        embedding_model (str): Embedding model identifier (EmbeddingModel enum value).
                               "openai"           → OpenAI embeddings
                               "nomic-embed-text" → Ollama nomic-embed-text

    Returns:
        A LangChain retrieval chain ready to call with {"input": ..., "chat_history": ...}
    """
    llm = get_llm(model)

    # Use the vectorstore that matches the chosen embedding model
    vs = get_vectorstore(embedding_model)
    retriever = vs.as_retriever(search_kwargs={"k": 2})

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain