import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import fitz  # PyMuPDF

# --- 1. SETUP ---
st.set_page_config(page_title="Medical RAG", layout="wide")

@st.cache_resource
def init_models():
    client = chromadb.PersistentClient(path="./my_ai_db")
    collection = client.get_or_create_collection(name="research_papers")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    genai.configure(api_key="AIzaSyDKh2DUE00P1UOALxWHYIGW4MsJvmNoKHw")
    llm = genai.GenerativeModel('gemini-2.5-flash-lite')
    return client, collection, embed_model, llm

client, collection, embed_model, llm = init_models()

# --- 2. SIDEBAR: PDF UPLOADER (Input #1) ---
# with st.sidebar:
#     st.header("Settings & Upload")
#     uploaded_file = st.file_uploader("Upload a new Research Paper (PDF)", type="pdf")
    
#     if uploaded_file:
#         if st.button("Process & Add to Brain"):
#             with st.spinner("Chunking and Embedding..."):
#                 # Read PDF directly from the uploader
#                 doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
#                 for page_num, page in enumerate(doc):
#                     text = page.get_text()
#                     # Basic chunking: split by 500 chars
#                     chunks = [text[i:i+500] for i in range(0, len(text), 400)] # 100 char overlap
                    
#                     for i, chunk in enumerate(chunks):
#                         vec = embed_model.encode([chunk]).tolist()
#                         collection.upsert(
#                             ids=[f"{uploaded_file.name}_{page_num}_{i}"],
#                             embeddings=vec,
#                             documents=[chunk],
#                             metadatas=[{"source": uploaded_file.name, "page": page_num + 1}]
#                         )
#             st.success(f"Added {uploaded_file.name} to the database!")

with st.sidebar:
    st.header("Database Stats")
    # This tells you how many total chunks are saved
    try:
        current_count = collection.count()
        st.write(f"Total Chunks in Brain: {current_count}")
    except Exception:
        st.write("Total Chunks in Brain: 0 (Initializing...)")
    
    if st.button("Clear Database"):
        try:
            # 1. Delete the old one
            client.delete_collection("research_papers")
        except Exception:
            # If it's already gone, don't crash
            pass
        
        # 2. Immediately recreate it so the rest of the app doesn't crash
        st.session_state.collection = client.get_or_create_collection(name="research_papers")
        
        st.success("Database wiped! Re-upload your papers.")
        st.rerun()

    st.header("Upload New Paper")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    
    if uploaded_file and st.button("Process & Add"):
        with st.spinner("Processing..."):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            new_chunks_count = 0
            
            for page_num, page in enumerate(doc):
                text = page.get_text()
                # Split text into 500-character chunks
                chunks = [text[i:i+500] for i in range(0, len(text), 400)]
                
                for i, chunk in enumerate(chunks):
                    # CRITICAL: Use a unique ID that includes filename AND page AND chunk index
                    # This prevents Paper 2 from accidentally overwriting Paper 1
                    unique_id = f"{uploaded_file.name}_p{page_num}_c{i}"
                    
                    vec = embed_model.encode([chunk]).tolist()
                    collection.upsert(
                        ids=[unique_id],
                        embeddings=vec,
                        documents=[chunk],
                        metadatas=[{"source": uploaded_file.name, "page": page_num + 1}]
                    )
                    new_chunks_count += 1
            
            st.success(f"Added {new_chunks_count} new chunks from {uploaded_file.name}!")
            st.rerun() # Refresh to update the count

# --- 3. MAIN CHAT: (Input #2) ---
st.title("🩺 Research AI")
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User types here!
if user_prompt := st.chat_input("Ask a question about your uploaded papers..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # RAG Logic
    q_vec = embed_model.encode([user_prompt]).tolist()
    results = collection.query(query_embeddings=q_vec, n_results=10)
    
    context = "\n".join(results['documents'][0])
    
    with st.chat_message("assistant"):
        final_prompt = f"Use ONLY this context: {context}\n\nQuestion: {user_prompt}"
        response = llm.generate_content(final_prompt)
        st.markdown(response.text)
        
        with st.expander("Sources Cited"):
            for meta in results['metadatas'][0]:
                st.write(f"📄 {meta['source']} (Page {meta['page']})")
    
    st.session_state.messages.append({"role": "assistant", "content": response.text})