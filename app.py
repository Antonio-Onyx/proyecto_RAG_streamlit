import os
import tempfile
from dotenv import load_dotenv
from client_llm_logic import get_llm_client
from RAG_modules_logic import process_document

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import streamlit as st

load_dotenv()

st.title(":rainbow[RAG]")
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


llm = ChatGroq(
    model=MODEL_TO_USE,
    api_key=os.getenv("GROQ_API_KEY"),
    temperature = 0.8,
    max_tokens=None,
    max_retries=2,
)

# config side bar to upload documents
with st.sidebar:
    st.header("📄 Your Documents")
    uploaded_files = st.file_uploader("Upload your documents here", type=["pdf", "txt"])

    if uploaded_files:
        # we need create a temporal directory to store the uploaded files
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_files.read())
            # save the path of the temporal file
            tmp_file_path = tmp_file.name

        st.success(f"Document '{uploaded_files.name}' loaded locally")

        # we need to check if the 'db' is in the backpack
        if "db" not in st.session_state or st.session_state.get("last_uploaded") != uploaded_files.name:
            with st.spinner("Processing document..."):
                try:
                    db = process_document(tmp_file_path)

                    st.session_state.db = db
                    st.session_state.last_uploaded = uploaded_files.name
                    st.success("Vectorial database created successfully.")
                except Exception as e:
                    os.remove(tmp_file_path)
        else:
            st.info("Data already loaded to chat")

        # here we will process the document with RAG logic


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
        
        # check if RAG exists
        if "db" in st.session_state:
            db = st.session_state.db
            retriever = db.as_retriever(search_kwargs={"k":5})

            template = """ Contesta la pregunta solo basandote en el contexto proporcionado:
            {context}

            Pregunta: {question}
            """

            prompt_template = ChatPromptTemplate.from_template(template)

            rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt_template
                | llm
                | StrOutputParser()
            )

            stream_handler = rag_chain.stream(prompt)
        
        else:
            simple_chain = llm | StrOutputParser()
            stream_handler = simple_chain.stream(prompt)

        try:
            for chunk in stream_handler:
                full_response += chunk
                message_placeholder.markdown(full_response + " ")
            
            message_placeholder.markdown(full_response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
            })
        except Exception as e:
            st.error(f"Error: {e}")
