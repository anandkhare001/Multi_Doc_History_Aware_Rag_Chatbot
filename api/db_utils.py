"""
This module provides utility functions for interacting with the SQLite database used in the RAG application.
It includes functions to create necessary tables, manage chat logs, and handle document records.
"""

import sqlite3
from datetime import datetime
import uuid

DB_NAME = "rag_app.db"


# Database Connection
def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


# Database Table Creation
def create_application_logs():
    """
    Create the application_logs table to store chat session logs.
    Fields:
      - id: Primary key
      - session_id: ID representing the chat session
      - user_query: User's input text
      - gpt_response: GPT generated response
      - model: Model used for the response
      - created_at: Timestamp of the log entry
    """
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS application_logs
    (id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    user_query TEXT,
    gpt_response TEXT,
    model TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()


def create_document_store():
    """
    Create the document_store table to store metadata about uploaded documents.
    Fields:
      - id: Primary key
      - filename: Name of the uploaded file
      - upload_timestamp: When the file was uploaded
    """
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS document_store
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     filename TEXT,
                     upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()


# Chat Log Management

def insert_application_logs(session_id, user_query, gpt_response, model):
    conn = get_db_connection()
    conn.execute(
        'INSERT INTO application_logs (session_id, user_query, gpt_response, model) VALUES (?, ?, ?, ?)',
        (session_id, user_query, gpt_response, model)
    )
    conn.commit()
    conn.close()


def get_chat_history(session_id):
    """
    Retrieve chat history formatted for LangChain (role/content dicts).
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT user_query, gpt_response FROM application_logs WHERE session_id = ? ORDER BY created_at',
        (session_id,)
    )
    messages = []
    for row in cursor.fetchall():
        messages.extend([
            {"role": "human", "content": row['user_query']},
            {"role": "ai", "content": row['gpt_response']}
        ])
    conn.close()
    return messages


# --- NEW: Get all distinct sessions with a preview label ---
def get_all_sessions():
    """
    Retrieve all distinct chat sessions ordered by their first message timestamp descending.

    Returns:
        List of dicts with:
          - session_id (str)
          - preview (str): the first user message in the session, truncated to 60 chars
          - created_at (str): ISO timestamp of first message in session
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT session_id,
               user_query AS preview,
               MIN(created_at) AS created_at
        FROM application_logs
        GROUP BY session_id
        ORDER BY created_at DESC
    ''')
    rows = cursor.fetchall()
    conn.close()

    sessions = []
    for row in rows:
        preview = row['preview'] or "Untitled Chat"
        if len(preview) > 60:
            preview = preview[:57] + "..."
        sessions.append({
            "session_id": row['session_id'],
            "preview": preview,
            "created_at": row['created_at']
        })
    return sessions


# --- NEW: Get full message list for a session (for Streamlit display) ---
def get_session_messages(session_id):
    """
    Retrieve all chat messages for a session as a list suitable for st.session_state.messages.

    Returns:
        List of dicts with 'role' ('user' or 'assistant') and 'content' keys,
        or None if session does not exist.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT user_query, gpt_response FROM application_logs WHERE session_id = ? ORDER BY created_at',
        (session_id,)
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return None

    messages = []
    for row in rows:
        messages.append({"role": "user", "content": row['user_query']})
        messages.append({"role": "assistant", "content": row['gpt_response']})
    return messages


# Document Record Management

def insert_document_record(filename):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO document_store (filename) VALUES (?)', (filename,))
    file_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return file_id


def delete_document_record(file_name):
    conn = get_db_connection()
    conn.execute('DELETE FROM document_store WHERE filename = ?', (file_name,))
    conn.commit()
    conn.close()
    return True


def get_all_documents():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, filename, upload_timestamp FROM document_store ORDER BY upload_timestamp DESC')
    documents = cursor.fetchall()
    conn.close()
    return [dict(doc) for doc in documents]


def get_document_details(file_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, filename, upload_timestamp FROM document_store WHERE id = ? ORDER BY upload_timestamp DESC",
        (file_id,)
    )
    documents = cursor.fetchall()
    conn.close()
    return [dict(doc) for doc in documents]


# Initialize the database tables
create_application_logs()
create_document_store()
