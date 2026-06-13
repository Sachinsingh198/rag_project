import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_document(file_path):
    document_loader = TextLoader(file_path)

    raw_document = document_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 30,
        length_function = len
    )

    chunks = text_splitter.split_documents(raw_document)

    print(f"Successfully split document into {len(chunks)} chunks\n")

    for i, chunk in enumerate(chunks[:3]):
        print(f"=== CHUNK {i + 1} ===")
        print(chunk.page_content)
        print(f"Metadata associated: {chunk.metadata}\n")


if __name__ == "__main__":
    sample_file = "sample.txt"
    if os.path.exists(sample_file):
        process_document(sample_file)
    else:
        print(f"Error: Please create a file named '{sample_file}' in this folder first!")