import streamlit as st
from api_utils import get_api_response, list_sessions


def display_chat_interface():
    """
    Display the main chat interface.

    - Renders all messages in the current session.
    - Accepts new user input and streams a response from the API.
    - After every new AI reply, refreshes the sidebar session list so the
      new session (or updated session) appears immediately.
    """

    # Show a subtle indicator of which session is active
    if st.session_state.get("session_id"):
        st.caption(f"Session: `{st.session_state.session_id}`")
    else:
        st.caption("New conversation — session will be created on your first message.")

    # Render existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new user input
    if prompt := st.chat_input("Ask anything about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Generating response..."):
            response = get_api_response(
                prompt,
                st.session_state.session_id,
                st.session_state.model
            )

            if response:
                # Persist the session_id returned by the backend
                st.session_state.session_id = response.get("session_id")

                answer = response["answer"]
                st.session_state.messages.append({"role": "assistant", "content": answer})

                with st.chat_message("assistant"):
                    st.markdown(answer)

                with st.expander("Details"):
                    st.subheader("Generated Answer")
                    st.code(answer)
                    st.subheader("Model Used")
                    st.code(response["model"])
                    st.subheader("Session ID")
                    st.code(response["session_id"])

                # Refresh the sidebar session list so this conversation appears
                st.session_state.sessions = list_sessions()

            else:
                st.error("Failed to get a response from the API. Please try again.")
