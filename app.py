import os
from dotenv import load_dotenv
from client_llm_logic import get_llm_client

import streamlit as st

load_dotenv()

st.title("RAG")
st.subheader("chat with your own documents")

API_HOST = os.getenv("API_HOST", "groq").lower()

if API_HOST == "groq":
    MODEL_TO_USE = "openai/gpt-oss-120b"
elif API_HOST == "gemini":
    MODEL_TO_USE = "gemini-2.5-flash"


try:
    client = get_llm_client()
    print(f"Using {API_HOST} client with model: {MODEL_TO_USE}")
    #full_story = ""
    #for token in response:
    #    full_story += token
    #    print(token, end="", flush=True)
except Exception as e:
    print(f"\n[ERROR] an error occured: {e}")


# inizializate chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat meesages from history and on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# accept user input
if prompt := st.chat_input("What is up?"):
    # add use message to chat history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
    })
    # display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        assistant_response = client.generate_content(prompt, MODEL_TO_USE)
        for token in assistant_response:
            full_response += token
            message_placeholder.markdown(full_response)
        message_placeholder.markdown(full_response)
    # add assistant response to chat history
    st.session_state.messages.append({
        "role":"assistant",
        "content":full_response,
    })