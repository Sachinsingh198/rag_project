import os
import yt_dlp
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama

# =====================================================================
# 1. FETCH YOUTUBE TRANSCRIPT
# =====================================================================
video_id = "J5_-l7WIO_w"
video_url = f"https://www.youtube.com/watch?v={video_id}"

def fetch_youtube_transcript_native(url: str) -> str:
    print("Step 1: Fetching YouTube transcript using yt-dlp...")
    ydl_opts = {'writeallsubtitles': True, 'skip_download': True, 'quiet': True, 'no_warnings': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            subtitles = info.get('subtitles', {})
            automatic_captions = info.get('automatic_captions', {})
            
            for lang in ['hi', 'en']:
                if lang in subtitles: return extract_text_from_subs(subtitles[lang])
            for lang in ['hi', 'en']:
                if lang in automatic_captions: return extract_text_from_subs(automatic_captions[lang])
            return ""
    except Exception as e:
        print(f"Error gathering transcript: {e}")
        return ""

def extract_text_from_subs(sub_info_list) -> str:
    text_pieces = []
    json_formats = [f for f in sub_info_list if f.get('ext') == 'json3']
    if json_formats:
        import urllib.request, json
        with urllib.request.urlopen(json_formats[0]['url']) as response:
            data = json.loads(response.read().decode())
            for event in data.get('events', []):
                if 'segs' in event:
                    for seg in event['segs']: text_pieces.append(seg['utf8'])
        return "".join(text_pieces)
    return ""

raw_text = fetch_youtube_transcript_native(video_url)

# =====================================================================
# 2. RUN PIPELINE IF TEXT EXISTS
# =====================================================================
if raw_text and len(raw_text.strip()) > 50:
    print(f"Success! Retrieved {len(raw_text)} characters.")
    
    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.create_documents([raw_text])
    print(f"Created {len(chunks)} chunks.")

    # Embedding & FAISS Store
    print("Initializing embedding model and building FAISS store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Local LLM Execution
    print("Connecting to local Ollama instance...")
    # NOTE: Since this script runs on Colab's cloud GPU now, we can use the full 'llama3' model!
    llm = ChatOllama(model="llama3", temperature=0.3)

    user_query = "What is the main summary of this video?"
    matched_docs = vector_store.similarity_search(user_query, k=2)
    context = "\n\n".join([doc.page_content for doc in matched_docs])

    final_prompt = f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"
    answer = llm.invoke(final_prompt)

    print("\n================== LLM RESPONSE ==================")
    print(answer.content)
    print("==================================================")
else:
    print("❌ Failed to grab a valid transcript track.")