import streamlit as st
import os
import yt_dlp
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama

# Streamlit App Title & Config
st.set_page_config(page_title="YouTube Video RAG Assistant", layout="centered")
st.title("🎥 YouTube Transcript RAG Assistant")
st.write("Analyze any YouTube video using a local Llama 3 model on Google Colab's GPU!")

# Input Form
video_url = st.text_input("Paste YouTube Video URL here:", "https://www.youtube.com/watch?v={your_video_id}")
user_query = st.text_input("Ask a question about the video:", "What is the main summary of this video?")

# Helper parsing logic
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

if st.button("Run AI Analysis"):
    if not video_url:
        st.warning("Please provide a valid YouTube link.")
    else:
        with st.spinner("Step 1: Downloading transcript using browser emulation..."):
            ydl_opts = {'writeallsubtitles': True, 'skip_download': True, 'quiet': True, 'no_warnings': True}
            raw_text = ""
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(video_url, download=False)
                    subtitles = info.get('subtitles', {})
                    automatic_captions = info.get('automatic_captions', {})
                    for lang in ['hi', 'en']:
                        if lang in subtitles: raw_text = extract_text_from_subs(subtitles[lang])
                    if not raw_text:
                        for lang in ['hi', 'en']:
                            if lang in automatic_captions: raw_text = extract_text_from_subs(automatic_captions[lang])
            except Exception as e:
                st.error(f"Error gathering transcript: {e}")

        if raw_text:
            st.success("Transcript fetched successfully!")
            
            # 2. Chunking
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.create_documents([raw_text])
            
            # 3. Embedding Vector Store
            with st.spinner("Step 2: Vectorizing text chunks into local memory store..."):
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
                vector_store = FAISS.from_documents(chunks, embeddings)
            st.success("FAISS Database built successfully!")

            # 4. LLM Query
            with st.spinner("Step 3: Querying Llama 3 via local cloud GPU context..."):
                llm = ChatOllama(model="llama3", temperature=0.3)
                matched_docs = vector_store.similarity_search(user_query, k=2)
                context = "\n\n".join([doc.page_content for doc in matched_docs])
                
                final_prompt = f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"
                answer = llm.invoke(final_prompt)
            
            # Render Response to Webpage UI
            st.markdown("### 🤖 LLM Response:")
            st.write(answer.content)
        else:
            st.error("Could not extract a valid transcript for this video link.")