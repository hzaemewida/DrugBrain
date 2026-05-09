import os
import streamlit as st
import numpy as np
from PIL import Image
import gc

# استيراد المكتبات مع تحديث مسارات الـ Text Splitter الجديدة
try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import CharacterTextSplitter # المسار الجديد
    from langchain_groq import ChatGroq
except ImportError as e:
    st.error(f"❌ خطأ في المكتبات: {e}")
    st.info("تأكد من تحديث ملف requirements.txt وعمل Reboot للتطبيق من لوحة التحكم.")
    st.stop()

# ====== إعدادات الصفحة ======
st.set_page_config(
    page_title="Drugbrain Intelligence OS",
    layout="wide",
    page_icon="🛸"
)

# ====== التصميم (CSS) ======
st.markdown("""
    <style>
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .animated-title {
        background: linear-gradient(270deg, #ff007f, #7f00ff, #00d2ff, #3a7bd5);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 5s ease infinite;
        text-align: center;
        font-size: 3rem;
        font-weight: 900;
        margin-bottom: 20px;
    }
    .report-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        border-right: 10px solid #7f00ff;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        color: #1a1a1a;
        direction: rtl;
        text-align: right;
        margin-top: 15px;
    }
    </style>
    <h1 class="animated-title">🛸 Drugbrain Intelligence OS 🧬</h1>
""", unsafe_allow_html=True)

# ====== الدوال الأساسية مع Caching ======
@st.cache_resource
def get_llm():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("❌ GROQ_API_KEY missing in Secrets!")
        st.stop()
    return ChatGroq(api_key=api_key, model_name="llama-3.3-70b-versatile", temperature=0.1)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_vector_store():
    embed_model = get_embeddings()
    index_name = "faiss_index_storage"
    
    # محاولة تحميل قاعدة البيانات لو موجودة لتوفير الرامات
    if os.path.exists(index_name):
        return FAISS.load_local(index_name, embed_model, allow_dangerous_deserialization=True)

    # لو مش موجودة، هنقرأ الكتب (اتأكد إن الأسماء صحيحة في جيت هاب)
    books = ["Clinical Pharmacology Made Incredibly Easy (3rd Ed.).pdf", "Book_2.pdf", "Book_3.pdf"]
    all_docs = []
    
    for book in books:
        if os.path.exists(book):
            loader = PyPDFLoader(book)
            all_docs.extend(loader.load())
    
    if not all_docs:
        return None

    # تقسيم النصوص
    text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    docs = text_splitter.split_documents(all_docs)
    
    # بناء الفهرس وحفظه محلياً
    vector_store = FAISS.from_documents(docs, embed_model)
    vector_store.save_local(index_name)
    return vector_store

@st.cache_resource
def get_ocr():
    import easyocr
    return easyocr.Reader(['en'], gpu=False)

def ask_ai(llm, vector_store, query):
    docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([d.page_content for d in docs])
    prompt = f"استخدم السياق التالي: {context}\n\nأجب على: {query}\n\nاجعل الإجابة طبية دقيقة وباللهجة المصرية."
    return llm.invoke(prompt).content

# ====== واجهة المستخدم ======
def main():
    llm = get_llm()
    
    with st.spinner("🧠 جاري تحميل القاعدة الطبية..."):
        vector_store = get_vector_store()

    if not vector_store:
        st.error("⚠️ لم يتم العثور على ملفات الـ PDF. ارفع الكتب وتأكد من أسمائها.")
        return

    tabs = st.tabs(["👁️ الروشتات", "💬 استفسار", "⚠️ تعارضات", "🩸 تحاليل"])

    with tabs[0]:
        file = st.file_uploader("ارفع صورة الروشتة", type=['jpg', 'png', 'jpeg'])
        if file and st.button("تحليل"):
            with st.spinner("جاري القراءة..."):
                reader = get_ocr()
                img = np.array(Image.open(file))
                text = " ".join(reader.readtext(img, detail=0))
                del img # تنظيف رام
                gc.collect()
                
                res = ask_ai(llm, vector_store, f"حلل الروشتة دي: {text}")
                st.markdown(f"<div class='report-card'>{res}</div>", unsafe_allow_html=True)

    with tabs[1]:
        q = st.text_input("اسأل عن دواء أو مرض:")
        if q and st.button("بحث"):
            res = ask_ai(llm, vector_store, q)
            st.markdown(f"<div class='report-card'>{res}</div>", unsafe_allow_html=True)

    with tabs[2]:
        drugs = st.text_area("ادخل الأدوية:")
        if drugs and st.button("فحص"):
            res = ask_ai(llm, vector_store, f"هل في تعارض بين {drugs}؟")
            st.markdown(f"<div class='report-card'>{res}</div>", unsafe_allow_html=True)

    with tabs[3]:
        lab = st.file_uploader("ارفع صورة التحليل", type=['jpg', 'png', 'jpeg'], key="lab_up")
        if lab and st.button("قراءة التحليل"):
            with st.spinner("جاري التحليل..."):
                reader = get_ocr()
                text = " ".join(reader.readtext(np.array(Image.open(lab)), detail=0))
                res = ask_ai(llm, vector_store, f"حلل نتائج التحليل دي: {text}")
                st.markdown(f"<div class='report-card'>{res}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
