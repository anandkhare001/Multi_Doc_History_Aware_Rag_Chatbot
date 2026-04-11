import streamlit as st
from sidebar import display_sidebar
from chat_interface import display_chat_interface

st.set_page_config(page_title="Multi-Doc RAG Chatbot", layout="wide")
st.title("Multi-Doc RAG Chatbot")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "model" not in st.session_state:
    st.session_state.model = "gpt-3.5-turbo"

# Display the sidebar (chat history + document management + model selector)
display_sidebar()

# Display the main chat interface
display_chat_interface()
