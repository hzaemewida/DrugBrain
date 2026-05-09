import os
import streamlit as st
import numpy as np
from PIL import Image
import gc

# استيراد المكتبات الأساسية مع معالجة الأخطاء
try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import CharacterTextSplitter
    from langchain_groq import ChatGroq
except ImportError:
    st.error("⚠️ خطأ في المكتبات! تأكد من تحديث ملف requirements.txt")
    st.stop()

# ====== 1. إعدادات الصفحة والتصميم (UI) ======
st.set_page_config(page_title="Drugbrain Intelligence OS", layout="wide", page_icon="🛸")

st.markdown("""
    <style>
    @keyframes gradient-shift { 0% {background-position:0% 50%} 50% {background-position:100% 50%} 100% {background-position:0% 50%} }
    .animated-title {
        background: linear-gradient(270deg, #ff007f, #7f00ff, #00d2ff, #3a7bd5);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 5s ease infinite;
        text-align: center; font-size: 3.2rem; font-weight: 900; margin-bottom: 20px;
    }
    .report-card {
        background: white; padding: 25px; border-radius: 15px; 
        border-right: 10px solid #7f00ff; box-shadow: 0 4px 15px rgba(0,0,0,0.1); 
        color: #1a1a1a; direction: rtl; text-align: right; margin-bottom: 20px;
        font-size: 18px; line-height: 1.6;
    }
    .stButton>button { background: linear-gradient(90deg, #7f00ff, #ff007f); color: white; border-radius: 10px; border:none; padding: 10px 20px; }
    </style>
    <h1 class="animated-title">🛸 Drugbrain Intelligence OS 🧬</h1>
""", unsafe_allow_html=True)

# ====== 2. الدوال الأساسية (الموفرة للذاكرة) ======
@st.cache_resource
def get_llm():
    return ChatGroq(api_key=st.secrets["GROQ_API_KEY"], model_name="llama-3.3-70b-versatile", temperature=0.1)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_vector_store():
    embed_model = get_embeddings()
    index_path = "faiss_index_v_final"
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embed_model, allow_dangerous_deserialization=True)
    
    # تحميل الكتب لو الفهرس مش موجود
    books = ["Clinical Pharmacology Made Incredibly Easy (3rd Ed.).pdf"] 
    all_docs = []
    for b in books:
        if os.path.exists(b): all_docs.extend(PyPDFLoader(b).load())
    
    if not all_docs: return None
    v_store = FAISS.from_documents(CharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(all_docs), embed_model)
    v_store.save_local(index_path)
    return v_store

@st.cache_resource
def get_ocr():
    import easyocr
    return easyocr.Reader(['en'], gpu=False)

def ask_ai(llm, v_store, query, is_table=False):
    docs = v_store.similarity_search(query, k=3)
    context = "\n".join([d.page_content for d in docs])
    table_note = "هام: استخدم جداول Markdown المنظمة (الدواء | الجرعة | ملاحظات) إذا كان الرد يحتوي على أدوية." if is_table else ""
    
    prompt = f"""استخدم المراجع: {context}\nالسؤال: {query}\n{table_note}\nأجب باللهجة المصرية العامية كخبير صيدلي متمكن."""
    return llm.invoke(prompt).content

# ====== 3. التطبيق الرئيسي (Main) ======
def main():
    # --- نظام التنظيف الذاتي (Auto-Reboot Logic) ---
    if 'use_count' not in st.session_state: st.session_state.use_count = 0
    st.session_state.use_count += 1
    if st.session_state.use_count % 5 == 0: gc.collect() # تنظيف كل 5 عمليات

    # --- شريط الجانب (Sidebar) ---
    with st.sidebar:
        st.header("⚙️ إدارة النظام")
        if st.button("♻️ تنشيط الذاكرة (Reboot)"):
            st.cache_resource.clear()
            gc.collect()
            st.rerun()
        st.info("استخدم الزرار ده لو حسيت ببطء أو البرنامج هنج.")

    # تحضير الموارد
    llm = get_llm()
    v_store = get_vector_store()
    if not v_store:
        st.error("⚠️ المراجع الطبية غير موجودة!")
        return

    # --- تقسيم الخانات (4 Tabs) ---
    tab1, tab2, tab3, tab4 = st.tabs(["👁️ الروشتات", "💬 استفسار طبي", "⚠️ التعارضات", "🩸 التحاليل"])

    # الخانة 1: الروشتات
    with tab1:
        f1 = st.file_uploader("ارفع صورة الروشتة:", type=['jpg','png','jpeg'], key="u1")
        if f1 and st.button("🚀 تحليل الروشتة"):
            with st.spinner("👀 جاري القراءة..."):
                reader = get_ocr()
                img = np.array(Image.open(f1))
                text = " ".join(reader.readtext(img, detail=0))
                del img; gc.collect()
            with st.spinner("🧠 جاري كتابة التقرير..."):
                ans = ask_ai(llm, v_store, f"حلل الروشتة واستخرج الأدوية: {text}", is_table=True)
                st.markdown(f"<div class='report-card'><h3>🩺 تقرير الروشتة:</h3>{ans}</div>", unsafe_allow_html=True)

    # الخانة 2: استفسار طبي
    with tab2:
        q = st.text_input("اسأل عن أي دواء أو حالة طبية:")
        if q and st.button("🔍 ابحث الآن"):
            with st.spinner("🧠 جاري البحث..."):
                ans = ask_ai(llm, v_store, q)
                st.markdown(f"<div class='report-card'>{ans}</div>", unsafe_allow_html=True)

    # الخانة 3: التعارضات
    with tab3:
        drugs = st.text_area("اكتب أسماء الأدوية (مثال: Aspirin, Warfarin):")
        if drugs and st.button("🚨 فحص التعارضات"):
            with st.spinner("🚨 جاري الفحص..."):
                ans = ask_ai(llm, v_store, f"هل يوجد تعارض بين: {drugs}؟", is_table=True)
                st.markdown(f"<div class='report-card'>{ans}</div>", unsafe_allow_html=True)

    # الخانة 4: التحاليل
    with tab4:
        f2 = st.file_uploader("ارفع صورة التحليل:", type=['jpg','png','jpeg'], key="u2")
        if f2 and st.button("🧬 تحليل النتائج"):
            with st.spinner("🔍 جاري فحص التحليل..."):
                reader = get_ocr()
                img = np.array(Image.open(f2))
                text = " ".join(reader.readtext(img, detail=0))
                del img; gc.collect()
            with st.spinner("🩸 جاري التقييم..."):
                ans = ask_ai(llm, v_store, f"حلل نتائج التحليل دي: {text}")
                st.markdown(f"<div class='report-card'><h3>🩸 تحليل المختبر:</h3>{ans}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
