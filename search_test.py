from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DB_DIR = 'chroma_db'

embedding_model = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

db = Chroma(persist_directory = DB_DIR, embedding_function = embedding_model)

query = input("Enter your query: ")

results = db.similarity_search(query, k=2)

print("\n--- SEARCH RESULTS ---")
for i, doc in enumerate(results):
    print(f"\n[Match {i+1}]")
    print(f"Source: {doc.metadata.get('source')}")
    print(f"Content: {doc.page_content}")

