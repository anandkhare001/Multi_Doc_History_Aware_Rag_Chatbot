from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import List
from langchain_core.documents import Document
import os
from chroma_utils import vectordb
from dotenv import load_dotenv

# Load environment variables from .env file, especially for the OpenAI API key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize the document retriever with search parameters (return top 2 results)
retriever = vectordb.as_retriever(search_kwargs={"k": 2})

# Output parser to convert output to string
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


# Creating the Retrieval-Augmented Generation Chain
# --------------------------------------------------
def get_rag_chain(model="gpt-3.5-turbo"):
    """
    Create and return a Retrieval-Augmented Generation (RAG) chain.

    This chain integrates a history-aware retriever that reformulates user questions
    to standalone questions based on chat history, and a question-answering chain
    that uses retrieved documents as context to answer the user query.

    Args:
        model (str): The identifier for the language model to use. Default is "gpt-3.5-turbo".

    Returns:
        A retrieval chain object that can be called to process user queries with
        context-aware retrieval and answer generation.
    """
    llm = ChatOpenAI(model=model)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain