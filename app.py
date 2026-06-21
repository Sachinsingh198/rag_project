import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DB_DIR = 'chroma_db'

def process_document(file_path):
    document_loader = TextLoader(file_path)

    raw_document = document_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 30,
        length_function = len
    )

    chunks = text_splitter.split_documents(raw_document)

    # 3. Initialize the free, local embedding model
    print("--- Initializing Embedding Model (this may take a moment on first run)... ---")
    embedding_model = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

    # 4. Create the Vector Database and save the chunks to disk
    print(f"--- Saving chunks to vector database at './{DB_DIR}'... ---")
    vector_db = Chroma.from_documents(
        documents = chunks, 
        embedding = embedding_model,  
        persist_directory = DB_DIR # This folder will save your data permanently
    )

    


if __name__ == "__main__":
    sample_file = "sample.txt"
    if os.path.exists(sample_file):
        process_document(sample_file)
    else:
        print(f"Error: Please create a file named '{sample_file}' in this folder first!")