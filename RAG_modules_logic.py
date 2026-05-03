import os
from dotenv import load_dotenv
load_dotenv()

import pymupdf4llm # to load pdf documents (instead of pypdf)
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client, Client


supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

def process_document(document: str) -> SupabaseVectorStore:
    """
    RAG chain logic to process document
    """
    chunks = text_splitter(document) #chunks -> embeddings
    embeddings = generate_embeddings() # embeddings -> vector store
    db = vector_store(embeddings=embeddings, documents=chunks) #vector store -> ids
    return db, embeddings

def text_splitter(document: str) -> list:

    md_text = pymupdf4llm.to_markdown(document)

    headers_to_split_on = [
        ("#", "title"),
        ("##", "section"),
        ("###", "subsection"),
    ]

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    md_chunks = md_splitter.split_text(md_text)

    recursive_splitter =  RecursiveCharacterTextSplitter(
        chunk_size=15000,
        chunk_overlap=200,
    )
    chunks = recursive_splitter.split_documents(md_chunks)

    return chunks

def generate_embeddings() -> HuggingFaceEmbeddings:
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return embeddings

def vector_store(embeddings: HuggingFaceEmbeddings, documents: list) -> SupabaseVectorStore:
    db = SupabaseVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        client=supabase,
        table_name='documents',
        query_name="match_documents",
        chunk_size=500,
    )
    return db

def load_existing_vector_store(embeddings: HuggingFaceEmbeddings) -> SupabaseVectorStore:
    """
    In case you want to load an existing vector store from db
    """
    return SupabaseVectorStore(
        embedding=embeddings,
        client=supabase,
        table_name='documents',
        query_name="match_documents",
    )