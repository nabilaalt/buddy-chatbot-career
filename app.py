import streamlit as st
import os
import nest_asyncio
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.memory import ChatMemoryBuffer

nest_asyncio.apply()

# ================= API KEY =================
GOOGLE_API_KEY = "AIzaSyA310pkbr0mg-m5agXKNTsIc3GaorJyheM"

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Career Buddy | Your Career Mentor",
    page_icon="💼",
    layout="wide"
)

# ================= PREMIUM UI CSS =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main {
    background-color: #f6f7fb;
}

/* Chat bubble */
.stChatMessage {
    border-radius: 18px;
    padding: 14px 16px;
    margin-bottom: 12px;
    max-width: 85%;
    box-shadow: 0 4px 14px rgba(0,0,0,0.04);
}

.stChatMessage.user {
    background-color: #e8efff;
    margin-left: auto;
}

.stChatMessage.assistant {
    background-color: #ffffff;
}

/* Input */
textarea {
    border-radius: 14px !important;
    padding: 12px !important;
}

/* Buttons */
button {
    border-radius: 14px !important;
    font-weight: 500 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff, #f1f3f8);
}
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1063/1063376.png", width=80)
    st.markdown("## Career Buddy")
    st.caption("Your personal career mentor")

    if st.button("🗑️ Reset Chat"):
        st.session_state.messages = []
        if "chat_engine" in st.session_state:
            st.session_state.chat_engine.reset()
        st.rerun()

    st.divider()
    if os.path.exists("data") and os.listdir("data"):
        st.success(f"{len(os.listdir('data'))} dokumen aktif")
    else:
        st.warning("Belum ada dokumen")

# ================= CORE SYSTEM =================
@st.cache_resource(show_spinner=False)
def load_rag_system():
    llm = GoogleGenAI(
        model="models/gemini-2.5-flash",
        api_key=GOOGLE_API_KEY,
        temperature=0.3
    )

    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    Settings.llm = llm
    Settings.embed_model = embed_model

    if not os.path.exists("data") or not os.listdir("data"):
        return None

    docs = SimpleDirectoryReader("./data").load_data()
    return VectorStoreIndex.from_documents(docs)

index = load_rag_system()

# ================= CHAT ENGINE =================
if "chat_engine" not in st.session_state and index:
    system_prompt = (
        "Kamu adalah Career Buddy, mentor karir profesional. "
        "Jawaban HARUS ringkas, mengalir seperti percakapan, dan berupa rangkuman. "
        "Gunakan maksimal 4–5 bullet point saja."
        "Jika tidak tahu jawabannya, kamu beri tahu apa yang kamu ketahui saja secara umum tapi masih relevan dengan yang dibahas ya"
        "Gausah mention sumber informasi. Cukup berbicara saja seperti teman ngobrol."
    )

    memory = ChatMemoryBuffer.from_defaults(token_limit=4096)
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        system_prompt=system_prompt
    )

# ================= HEADER =================
st.markdown("""
<h1 style="margin-bottom:4px;">Career Buddy</h1>
<p style="color:#6b7280; font-size:16px; margin-bottom:20px;">
Ngobrol santai soal CV, interview, dan strategi karir.
</p>
""", unsafe_allow_html=True)

# ================= QUICK SUGGESTIONS (STICKY FEEL) =================
cols = st.columns(3)
suggestions = [
    "Cara membuat CV ATS",
    "Tips interview kerja",
    "Cara negosiasi gaji"
]
for i, s in enumerate(suggestions):
    if cols[i].button(s):
        st.session_state.active_suggestion = s

st.divider()

# ================= CHAT HISTORY =================
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Halo! Aku **Buddy** 👋\nSiap bantu persiapan karirmu. Mau mulai dari mana?"
    }]

for msg in st.session_state.messages:
    avatar = "💼" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ================= INPUT =================
user_input = st.chat_input("Tanya apa aja tentang karir...")

if "active_suggestion" in st.session_state:
    user_input = st.session_state.active_suggestion
    del st.session_state.active_suggestion

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="💼"):
        if not index:
            full_response = "Dokumen referensi belum tersedia."
            st.markdown(full_response)
        else:
            response = st.session_state.chat_engine.stream_chat(user_input)
            full_response = st.write_stream(response.response_gen)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
