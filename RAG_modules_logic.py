from langchain_text_splitters import CharacterTextSplitter
import pymupdf4llm # to load pdf documents (instead of pypdf)
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

def process_document(document: str) -> FAISS:
    """
    RAG chain logic to process document
    """
    chunks = text_splitter(document) #chunks -> embeddings
    embeddings = generate_embeddings() # embeddings -> vector store
    db = vector_store(embeddings=embeddings, documents=chunks) #vector store -> ids
    return db

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
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

def vector_store(embeddings: HuggingFaceEmbeddings, documents: list) -> FAISS:
    dim_sample = embeddings.embed_query("Hello world")
    embedding_dim = len(dim_sample)
    index = faiss.IndexFlatL2(embedding_dim)

    vector_store_obj = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(), # al docs save in memory ram
        index_to_docstore_id={}
    )
    ids = vector_store_obj.add_documents(documents=documents)

    return vector_store_obj