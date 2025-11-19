from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

def process_document(document):
    """
    RAG chain logic to process document
    """
    text_splitter = CharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=500,
        add_start_index=True,
    )
    loader = PyPDFLoader(document)
    documents = loader.load()
    print(f"Document laoded, number of pages: {len(documents)}")

    # generate embeddings
    chunks = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


    # vector store
    embedding_dim = len(embeddings.embed_query(chunks[0].page_content))
    index = faiss.IndexFlatL2(embedding_dim)

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(), # al docs save in memory ram
        index_to_docstore_id={}
    )
    ids = vector_store.add_documents(documents=documents)

def text_splitter(document):
    text_splitter = CharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=500,
        add_start_index=True,
    )
    loader = PyPDFLoader(document)
    documents = loader.load()
    chunks = text_splitter.split_documents(documents)
    return chunks

def generate_embeddings(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

def vector_store(embeddings, documents):
    embedding_dim = len(embeddings.embed_query(chunks[0].page_content))
    index = faiss.IndexFlatL2(embedding_dim)

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(), # al docs save in memory ram
        index_to_docstore_id={}
    )
    ids = vector_store.add_documents(documents=documents)
    return ids

if __name__ == "__main__":
    process_document("./data/en_las_montañas_de_la_locura.pdf")