import os
import streamlit as st
import numpy as np
from PIL import Image
import gc

# استيراد المكتبات
try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import CharacterTextSplitter
    from langchain_groq import ChatGroq
except ImportError:
    st.error("❌ في مشكلة في المكتبات، تأكد من تحديث requirements.txt وعمل Reboot.")
    st.stop()

# ====== إعدادات الصفحة ======
st.set_page_config(page_title="Drugbrain Intelligence OS", layout="wide", page_icon="🛸")

# ====== التصميم (CSS) ======
st.markdown("""
    <style>
    .animated-title {
        background: linear-gradient(270deg, #ff007f, #7f00ff, #00d2ff, #3a7bd5);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 5s ease infinite;
        text-align: center; font-size: 3.2rem; font-weight: 900;
    }
    @keyframes gradient-shift { 0% {background-position:0% 50%} 50% {background-position:100% 50%} 100% {background-position:0% 50%} }
    .report-card { 
        background: white; padding: 20px; border-radius: 15px; 
        border-right: 10px solid #7f00ff; box-shadow: 0 4px 15px rgba(0,0,0,0.1); 
        color: #1a1a1a; direction: rtl; text-align: right; margin-bottom: 20px;
    }
    .stButton>button { background: linear-gradient(90deg, #7f00ff, #ff007f); color: white; border-radius: 10px; }
    </style>
    <h1 class="animated-title">🛸 Drugbrain Intelligence OS 🧬</h1>
""", unsafe_allow_html=True)

# ====== الدوال الأساسية المحسنة ======
@st.cache_resource
def get_llm():
    return ChatGroq(api_key=st.secrets["GROQ_API_KEY"], model_name="llama-3.3-70b-versatile", temperature=0.1)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_vector_store():
    embed_model = get_embeddings()
    index_path = "faiss_index_v2"
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embed_model, allow_dangerous_deserialization=True)
    
    books = ["Clinical Pharmacology Made Incredibly Easy (3rd Ed.).pdf", "Book_2.pdf", "Book_3.pdf"]
    all_docs = []
    for b in books:
        if os.path.exists(b): all_docs.extend(PyPDFLoader(b).load())
    
    if not all_docs: return None
    v_store = FAISS.from_documents(CharacterTextSplitter(chunk_size=1200, chunk_overlap=150).split_documents(all_docs), embed_model)
    v_store.save_local(index_path)
    return v_store

@st.cache_resource
def get_ocr():
    import easyocr
    return easyocr.Reader(['en'], gpu=False)

def ask_ai(llm, v_store, query, is_table=False):
    docs = v_store.similarity_search(query, k=3)
    context = "\n".join([d.page_content for d in docs])
    table_instruction = "إذا كان الرد يحتوي على قائمة أدوية، استخدم جداول Markdown المنظمة (الاسم، الجرعة، ملاحظات)." if is_table else ""
    
    prompt = f"""استخدم المراجع: {context}
    السؤال: {query}
    {table_instruction}
    أجب باللهجة المصرية العامية كطبيب خبير."""
    return llm.invoke(prompt).content

# ====== واجهة المستخدم (4 Tabs) ======
def main():
    llm = get_llm()
    with st.spinner("🧬 جاري تحضير القاعدة الطبية..."):
        v_store = get_vector_store()

    if not v_store:
        st.error("⚠️ ملفات الـ PDF غير موجودة!")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["👁️ الروشتات", "💬 استفسار طبي", "⚠️ التعارضات", "🩸 التحاليل"])

    # --- تاب 1: الروشتات ---
    with tab1:
        file1 = st.file_uploader("ارفع صورة الروشتة", type=['jpg','png','jpeg'], key="rx_up")
        if file1 and st.button("🚀 تحليل الروشتة"):
            with st.spinner("👀 جاري القراءة..."):
                reader = get_ocr()
                img = np.array(Image.open(file1))
                text = " ".join(reader.readtext(img, detail=0))
                del img; gc.collect()
                ans = ask_ai(llm, v_store, f"حلل الروشتة دي واستخرج الأدوية: {text}", is_table=True)
                st.markdown(f"<div class='report-card'><h3>🩺 تقرير الروشتة:</h3>{ans}</div>", unsafe_allow_html=True)

    # --- تاب 2: استفسار طبي ---
    with tab2:
        q = st.text_input("اسأل عن أي دواء أو حالة:")
        if q and st.button("🔍 بحث"):
            ans = ask_ai(llm, v_store, q)
            st.markdown(f"<div class='report-card'>{ans}</div>", unsafe_allow_html=True)

    # --- تاب 3: التعارضات ---
    with tab3:
        drugs = st.text_area("أدخل أسماء الأدوية لفحص التفاعل:")
        if drugs and st.button("🚨 فحص"):
            ans = ask_ai(llm, v_store, f"هل في تعارض بين {drugs}؟", is_table=True)
            st.markdown(f"<div class='report-card'>{ans}</div>", unsafe_allow_html=True)

    # --- تاب 4: التحاليل ---
    with tab4:
        file2 = st.file_uploader("ارفع صورة التحليل", type=['jpg','png','jpeg'], key="lab_up")
        if file2 and st.button("🧬 تحليل النتيجة"):
            with st.spinner("🔍 جاري فحص التحليل..."):
                reader = get_ocr()
                text = " ".join(reader.readtext(np.array(Image.open(file2)), detail=0))
                gc.collect()
                ans = ask_ai(llm, v_store, f"حلل نتائج التحليل دي وقولي القيم الغلط: {text}")
                st.markdown(f"<div class='report-card'><h3>🩸 نتائج التحليل:</h3>{ans}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
