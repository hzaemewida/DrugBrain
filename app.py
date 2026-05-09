import os
import streamlit as st
import numpy as np
from PIL import Image
import gc
from fpdf import FPDF # مكتبة توليد الـ PDF

# استيراد المكتبات الأساسية
try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import CharacterTextSplitter
    from langchain_groq import ChatGroq
except ImportError as e:
    st.error(f"❌ خطأ في المكتبات: {e}")
    st.stop()

# ====== إعدادات الصفحة والتصميم ======
st.set_page_config(page_title="Drugbrain Intelligence OS", layout="wide", page_icon="🛸")

st.markdown("""
    <style>
    .animated-title {
        background: linear-gradient(270deg, #ff007f, #7f00ff, #00d2ff, #3a7bd5);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 5s ease infinite;
        text-align: center; font-size: 3rem; font-weight: 900;
    }
    @keyframes gradient-shift { 0% {background-position:0% 50%} 50% {background-position:100% 50%} 100% {background-position:0% 50%} }
    .report-card { background: white; padding: 20px; border-radius: 15px; border-right: 10px solid #7f00ff; box-shadow: 0 4px 15px rgba(0,0,0,0.1); color: #1a1a1a; direction: rtl; text-align: right; }
    </style>
    <h1 class="animated-title">🛸 Drugbrain Intelligence OS 🧬</h1>
""", unsafe_allow_html=True)

# ====== دالة توليد ملف PDF (خفيفة جداً) ======
def create_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    # تنظيف النص من علامات الـ Markdown للجداول عشان يظهر صح في الـ PDF
    clean_text = text.replace("|", " ").replace("-", " ")
    pdf.multi_cell(0, 10, txt=clean_text.encode('latin-1', 'replace').decode('latin-1'))
    return pdf.output(dest='S').encode('latin-1')

# ====== الدوال الأساسية المحسنة (Caching) ======
@st.cache_resource
def get_llm():
    return ChatGroq(api_key=st.secrets["GROQ_API_KEY"], model_name="llama-3.3-70b-versatile", temperature=0.1)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_vector_store():
    embed = get_embeddings()
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embed, allow_dangerous_deserialization=True)
    # لو مش موجود حمل الملفات (تأكد من وجود الكتب)
    books = ["Clinical Pharmacology Made Incredibly Easy (3rd Ed.).pdf"] 
    docs = []
    for b in books:
        if os.path.exists(b): docs.extend(PyPDFLoader(b).load())
    if not docs: return None
    v_store = FAISS.from_documents(CharacterTextSplitter(chunk_size=1200, chunk_overlap=150).split_documents(docs), embed)
    v_store.save_local("faiss_index")
    return v_store

def ask_ai_pro(llm, vector_store, query):
    docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([d.page_content for d in docs])
    # التعديل هنا لطلب جدول
    prompt = f"""استخدم المراجع التالية: {context}
    أجب على السؤال: {query}
    مهم جداً: إذا كانت الإجابة تتضمن أدوية أو تعليمات، اعرضها في شكل 'Markdown Table' منظم.
    أجب باللهجة المصرية العامية المبسطة."""
    return llm.invoke(prompt).content

# ====== التطبيق الرئيسي ======
def main():
    llm = get_llm()
    v_store = get_vector_store()
    
    if not v_store:
        st.error("⚠️ المراجع غير موجودة!")
        return

    tab1, tab2 = st.tabs(["💬 استفسار ذكي", "👁️ تحليل صور"])

    with tab1:
        query = st.text_input("اسأل عن أي تفاعل دوائي أو حالة:")
        if query and st.button("بحث"):
            with st.spinner("🧠 جاري التحليل..."):
                ans = ask_ai_pro(llm, v_store, query)
                st.markdown(f"<div class='report-card'>{ans}</div>", unsafe_allow_html=True)
                
                # زرار التحميل الاحترافي
                pdf_data = create_pdf(ans)
                st.download_button("📥 تحميل التقرير كـ PDF", data=pdf_data, file_name="Drugbrain_Report.pdf", mime="application/pdf")

    with tab2:
        file = st.file_uploader("ارفع روشتة أو تحليل", type=['jpg','png','jpeg'])
        if file and st.button("تحليل الصورة"):
            import easyocr
            reader = easyocr.Reader(['en'], gpu=False)
            img = np.array(Image.open(file))
            text = " ".join(reader.readtext(img, detail=0))
            del img; gc.collect() # تنظيف رام
            
            ans = ask_ai_pro(llm, v_store, f"حلل هذا النص المستخرج من صورة: {text}")
            st.markdown(f"<div class='report-card'>{ans}</div>", unsafe_allow_html=True)
            
            pdf_data = create_pdf(ans)
            st.download_button("📥 تحميل التقرير كـ PDF", data=pdf_data, file_name="Medical_Analysis.pdf", mime="application/pdf")

if __name__ == "__main__":
    main()
