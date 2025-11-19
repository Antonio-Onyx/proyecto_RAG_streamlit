from langchain_text_splitters import CharacterTextSplitter
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
    text_splitter = CharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=500,
        add_start_index=True,
    )
    loader = PyPDFLoader(document)
    documents = loader.load()
    chunks = text_splitter.split_documents(documents)
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

if __name__ == "__main__":
    result = process_document("./data/en_las_montañas_de_la_locura.pdf")
    print(f"Type: {type(result)}")