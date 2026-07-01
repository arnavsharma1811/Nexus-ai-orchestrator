import streamlit as st
import requests
import json
from PIL import Image
import io

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Nexus AI Orchestrator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
    .stApp { background-color: #101415; color: #e0e3e5; }
    [data-testid="stSidebar"] { background-color: rgba(25, 28, 30, 0.6) !important; backdrop-filter: blur(20px) !important; border-right: 1px solid rgba(255,255,255,0.1) !important; }
    [data-testid="stSidebar"] * { color: #e0e3e5 !important; }
    .glass-card { background: rgba(25, 28, 30, 0.6); backdrop-filter: blur(20px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 0.75rem; padding: 1.5rem; margin-bottom: 1rem; }
    .glass-card h3, .glass-card h4, .glass-card p { color: #e0e3e5; }
    .primary-text { color: #adc6ff; }
    .stButton button { background-color: #adc6ff !important; color: #00285c !important; font-weight: 600 !important; border-radius: 0.75rem !important; border: none !important; transition: all 0.2s; }
    .stButton button:hover { box-shadow: 0 0 20px rgba(173, 198, 255, 0.3); transform: scale(1.02); }
    .stButton button:active { transform: scale(0.98); }
    .stTextArea textarea, .stTextInput input { background-color: rgba(255,255,255,0.05) !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 0.75rem !important; color: #e0e3e5 !important; }
    .stTextArea textarea:focus, .stTextInput input:focus { border-color: #adc6ff !important; box-shadow: 0 0 15px rgba(173, 198, 255, 0.15) !important; }
    .stFileUploader > div { background-color: rgba(255,255,255,0.05) !important; border: 2px dashed rgba(173, 198, 255, 0.3) !important; border-radius: 0.75rem !important; }
    .stSlider > div > div > div { background-color: #adc6ff !important; }
    .streamlit-expanderHeader { background-color: rgba(255,255,255,0.03) !important; border-radius: 0.5rem !important; color: #e0e3e5 !important; }
    .metric-box { background: rgba(173, 198, 255, 0.08); border-radius: 0.5rem; padding: 0.5rem 1rem; border-left: 3px solid #adc6ff; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(173, 198, 255, 0.2); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
st.sidebar.title("🧠 Nexus AI")
st.sidebar.caption("Orchestrator v2.4")
page = st.sidebar.radio("Navigate", ["RAG", "Agent", "Voice", "Multi-Modal"], index=0, label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.button("🚀 Deploy Agent", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.markdown("📊 System Status")
st.sidebar.markdown("📘 Documentation")

# ---------- API BASE URL (READ FROM SECRETS) ----------
API_BASE = st.secrets.get("API_BASE", "http://localhost:8000")

# ---------- PAGES ----------
def rag_page():
    st.markdown('<h2 class="primary-text">📚 Retrieval Interface</h2>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        query = st.text_area("Ask a question from your notes:", height=120, placeholder="Describe the information you need...")
        col1, col2 = st.columns([3, 1])
        with col1:
            top_k = st.slider("Top K chunks", 1, 20, 5, key="rag_topk")
        with col2:
            st.markdown(f"<p style='margin-top: 2rem; text-align: center;'>Chunks: <span style='color: #adc6ff;'>{top_k}</span></p>", unsafe_allow_html=True)
        search_clicked = st.button("🔍 Search", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if search_clicked and query:
        with st.spinner("Searching..."):
            try:
                res = requests.post(f"{API_BASE}/rag/query", json={"query": query, "top_k": top_k})
                if res.status_code == 200:
                    data = res.json()
                    st.success("Answer generated")
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown(f"**{data.get('answer', 'No answer')}**")
                    st.caption(f"Latency: {data.get('latency_ms', 0)} ms")
                    with st.expander("📄 Sources"):
                        for i, src in enumerate(data.get("sources", [])):
                            st.write(f"**Source {i+1}:** {src}")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error(f"Error: {res.status_code}")
            except Exception as e:
                st.error(f"Connection error: {e}")
    elif search_clicked:
        st.warning("Please enter a query.")

def agent_page():
    st.markdown('<h2 class="primary-text">🤖 Agent Orchestrator</h2>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        prompt = st.text_area("Describe the task for the agent:", height=120, placeholder="E.g., Analyze the potential infrastructure bottlenecks...")
        tools = st.multiselect("Tools", ["rag", "calculator", "web_search"], default=["rag"])
        run_clicked = st.button("▶ Run Agent", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if run_clicked and prompt:
        with st.spinner("Agent thinking..."):
            try:
                res = requests.post(f"{API_BASE}/agent/run", json={"prompt": prompt, "tools": tools})
                if res.status_code == 200:
                    data = res.json()
                    st.success("Agent response")
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown(f"**{data.get('answer', 'No answer')}**")
                    with st.expander("🧠 Trace"):
                        for step in data.get("trace", []):
                            st.write(f"**Step {step.get('step')}:** {step.get('thought')}")
                            st.write(f"*Action:* {step.get('action')} → *Observation:* {step.get('observation')}")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error(f"Error: {res.status_code}")
            except Exception as e:
                st.error(f"Connection error: {e}")
    elif run_clicked:
        st.warning("Please enter a prompt.")

def voice_page():
    st.markdown('<h2 class="primary-text">🎤 Voice Intelligence</h2>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        text = st.text_input("Type what you'd say:", placeholder="What is paging?")
        speak_clicked = st.button("🗣 Speak", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if speak_clicked and text:
        with st.spinner("Generating voice response..."):
            try:
                res = requests.post(f"{API_BASE}/voice/talk", json={"text": text})
                if res.status_code == 200:
                    data = res.json()
                    st.success("Voice response")
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown(f"**{data.get('text', 'No response')}**")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error(f"Error: {res.status_code}")
            except Exception as e:
                st.error(f"Connection error: {e}")
    elif speak_clicked:
        st.warning("Please enter some text.")

def multi_modal_page():
    st.markdown('<h2 class="primary-text">🖼️ Diagram Analyzer</h2>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        with st.form(key="multi_form"):
            uploaded_file = st.file_uploader("Drop an image (PNG, JPG, SVG)", type=["png", "jpg", "jpeg", "svg"])
            query = st.text_input("Question about the diagram (optional):", placeholder="What does this diagram show?")
            submitted = st.form_submit_button("🔍 Analyze")
        st.markdown('</div>', unsafe_allow_html=True)

    if submitted and uploaded_file is not None:
        with st.spinner("Analyzing..."):
            try:
                files = {"image": uploaded_file}
                data = {"query": query} if query and query.strip() else {}
                st.write(f"Sending query: '{query}'")
                res = requests.post(f"{API_BASE}/multi-modal/analyze", files=files, data=data)
                if res.status_code == 200:
                    data = res.json()
                    st.success("Analysis complete")
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown("**Extracted Text:**")
                    st.code(data.get("extracted_text", ""))
                    st.markdown("**Answer:**")
                    st.markdown(f"_{data.get('answer', 'No answer')}_")
                    st.caption(f"Confidence: {data.get('confidence', 0)}")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error(f"Error: {res.status_code} - {res.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")
    elif submitted:
        st.warning("Please upload an image.")

# ---------- ROUTING ----------
if page == "RAG":
    rag_page()
elif page == "Agent":
    agent_page()
elif page == "Voice":
    voice_page()
elif page == "Multi-Modal":
    multi_modal_page()

st.markdown("---")
st.caption("Nexus AI v2.4 | Built with Streamlit & FastAPI")