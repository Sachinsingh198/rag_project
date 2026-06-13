import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def process_document(file_path):

    ## 1. Load the documents
    print(f"Loading the doucument : {file_path}")
    loader = TextLoader(file_path);
    raw_document = loader.load()

    # 2. Initialize the text splitter
    # chunk_size: maximum characters per chunk (approx. 100-150 words)
    # chunk_overlap: keeps a small window of text from the previous chunk 
    #                so context isn't lost at the boundaries
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50, 
        length_function = len
    )

    # 3. Split the document into chunks
    chunks = text_splitter.split_documents(raw_document)

    # 4. Let's inspect the first couple of chunks to see what they look like
    print(f"Successfully split document into {len(chunks)} smaller chunks\n")

    for i, chunk in enumerate(chunks[:3]):
        print(f"=== CHUNK {i + 1} ===")
        print(chunk.page_content)
        print(f"Metadata associated: {chunk.metadata}\n")



if __name__ == "__main__" : 
    sample_file = "sample.txt";
    
    if(os.path.exists(sample_file)):
        process_document(sample_file)
    else :
        print(f"Error: Please create a file named '{sample_file}' in this folder first!")