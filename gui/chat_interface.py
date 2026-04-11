import streamlit as st
from api_utils import get_api_response


# Function: display_chat_interface
# -------------------------------
# Displays the chat interface using Streamlit, showing previous messages,
# handling new user input, and displaying responses from an API.
# It manages the chat state including messages, session IDs, and model usage.
def display_chat_interface():
    # Display chat history from the session state
    for message in st.session_state.messages:
        # Display each message with the appropriate chat role styling
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new user input from the chat input box
    if prompt := st.chat_input("Query:"):
        # Append user message to the session state messages
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display a spinner while waiting for API response
        with st.spinner("Generating response..."):
            # Call API to get response based on user prompt, session ID and model
            response = get_api_response(prompt, st.session_state.session_id, st.session_state.model)

            if response:
                # Update session ID if changed in response
                st.session_state.session_id = response.get('session_id')
                st.session_state.session_id = response.get('session_id')
                # Append the assistant's answer to the chat messages
                st.session_state.messages.append({"role": "assistant", "content": response['answer']})

                # Display the assistant's response message
                with st.chat_message("assistant"):
                    st.markdown(response['answer'])

                # Expandable section showing detailed info about the response
                with st.expander("Details"):
                    st.subheader("Generated Answer")
                    st.code(response['answer'])
                    st.subheader("Model Used")
                    st.code(response['model'])
                    st.subheader("Session ID")
                    st.code(response['session_id'])
            else:
                # Show error message if API fails to return a response
                st.error("Failed to get a response from the API. Please try again.")

