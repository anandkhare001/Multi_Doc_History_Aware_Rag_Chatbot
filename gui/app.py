import streamlit as st
from sidebar import display_sidebar
from chat_interface import display_chat_interface

# Set the title of the application
st.title("Multi-Doc RAG Chatbot")

# Initialize session state variables if they do not exist
# 'messages' will store the conversation history between the user and the chatbot
if "messages" not in st.session_state:
    st.session_state.messages = []

# 'session_id' will keep track of the current chat session's unique identifier
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# Display the sidebar where users can upload and manage documents
# and select the language model to use
# This function is imported from the sidebar module
# It handles UI and interactions related to document management and model selection
display_sidebar()

# Display the main chat interface where the user can interact with the chatbot
# This function is imported from the chat_interface module
# It handles the chat input, output, and interaction logic
display_chat_interface()