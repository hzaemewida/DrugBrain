import os
import streamlit as st
import numpy as np
from PIL import Image

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq

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
        font-size: 3.5rem;
        font-weight: 900;
        margin-bottom: 30px;
    }
    .report-card {
        background: linear-gradient(135deg, #ffffff 0%, #f3f4f6 100%);
        padding: 25px;
        border-radius: 20px;
        border-right: 8px solid #7f00ff;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        color: #1a1a1a;
        direction: rtl;
        text-align: right;
        font-size: 18px;
        line-height: 1.8;
        margin-top: 20px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #7f00ff, #ff007f);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 25px;
        font-size: 18px;
    }
    </style>
    <h1 class="animated-title">🛸 Drugbrain Intelligence OS 🧬</h1>
""", unsafe_allow_html=True)


# ====== تحميل النموذج اللغوي ======
@st.cache_resource
def get_llm():
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    if not api_key:
        st.error("❌ لم يتم العثور على GROQ_API_KEY. أضفه في Secrets.")
        st.stop()
    return ChatGroq(
        api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.1
    )


# ====== تحميل نموذج الـ Embeddings ======
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ====== بناء قاعدة البيانات من ملفات PDF ======
@st.cache_resource
def get_vector_store():
    embed_model = get_embeddings()
    # غير الأسماء دي لو ملفاتك مختلفة
    books_list = [
        "Clinical Pharmacology Made Incredibly Easy (3rd Ed.).pdf",
        "Book_2.pdf",
        "Book_3.pdf"
    ]

    all_docs = []
    for path in books_list:
        if os.path.exists(path):
            try:
                loader = PyPDFLoader(path)
                all_docs.extend(loader.load())
            except Exception as e:
                st.warning(f"⚠️ مشكلة في تحميل {path}: {e}")

    if not all_docs:
        return None

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(all_docs)
    return FAISS.from_documents(split_docs, embed_model)


# ====== تحميل OCR (تحميل كسول لتقليل استهلاك الرام) ======
@st.cache_resource
def get_ocr_reader():
    import easyocr
    return easyocr.Reader(['en'], gpu=False)


# ====== دالة الإجابة الذكية ======
def ask_smart_assistant(llm, vector_store, query):
    docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([d.page_content for d in docs])

    full_prompt = f"""
    أنت طبيب وصيدلي مصري خبير. استخدم المعلومات الطبية التالية للإجابة:
    {context}

    المطلوب:
    {query}
    """
    response = llm.invoke(full_prompt)
    return response.content


# ====== التطبيق الرئيسي ======
def main():
    llm = get_llm()

    with st.spinner("🧠 جاري تحميل القاعدة الطبية..."):
        vector_store = get_vector_store()

    if not vector_store:
        st.error("❌ لم يتم العثور على ملفات PDF. تأكد من رفعها داخل الريبو.")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "👁️ الروشتات",
        "💬 استفسار طبي",
        "⚠️ التعارضات",
        "🩸 التحاليل"
    ])

    # ======= تاب الروشتات =======
    with tab1:
        st.info("💡 ارفع صورة الروشتة:")
        up_file = st.file_uploader(
            "صورة الروشتة",
            type=['jpg', 'png', 'jpeg'],
            key="rx"
        )
        if up_file and st.button("🚀 تحليل الروشتة"):
            with st.spinner("👀 جاري تحميل محرك القراءة..."):
                reader = get_ocr_reader()
            with st.spinner("👀 جاري قراءة الروشتة..."):
                img = Image.open(up_file)
                img_np = np.array(img)
                results = reader.readtext(img_np, detail=0)
                raw_ocr_text = " ".join(results)

            with st.spinner("🧠 جاري التحليل الطبي..."):
                q = (
                    f"النص ده من روشتة مريض: '{raw_ocr_text}'. "
                    "استنتج الأدوية الصح، استخداماتها، التشخيص، "
                    "واقترح تحاليل طبية ضرورية لو الحالة تستدعي. "
                    "اكتب تقرير منظم باللهجة المصرية."
                )
                ans = ask_smart_assistant(llm, vector_store, q)
                st.markdown(
                    f"<div class='report-card'><h3>🩺 تقرير الروشتة:</h3>{ans}</div>",
                    unsafe_allow_html=True
                )

    # ======= تاب الاستفسار الطبي =======
    with tab2:
        q = st.text_input("اسأل عن أي دواء:")
        if q and st.button("🔍 بحث"):
            with st.spinner("📚 جاري البحث في المراجع..."):
                ans = ask_smart_assistant(
                    llm,
                    vector_store,
                    f"أجب باللهجة المصرية: {q}"
                )
                st.markdown(
                    f"<div class='report-card'>🤖 <b>الإجابة:</b><br>{ans}</div>",
                    unsafe_allow_html=True
                )

    # ======= تاب التعارضات =======
    with tab3:
        st.warning("⚡ أدخل الأدوية لمعرفة التفاعلات الخطيرة.")
        drugs_input = st.text_area("مثال: Aspirin, Warfarin")
        if drugs_input and st.button("🧪 فحص التعارضات"):
            with st.spinner("🚨 جاري البحث عن التفاعلات..."):
                q = (
                    f"ابحث عن التفاعلات الدوائية بين: {drugs_input}. "
                    "هل يوجد تعارض خطير؟ وما البديل؟ "
                    "الرد باللهجة المصرية."
                )
                ans = ask_smart_assistant(llm, vector_store, q)
                st.markdown(
                    f"<div class='report-card'><h3>⚠️ التفاعلات الدوائية:</h3>{ans}</div>",
                    unsafe_allow_html=True
                )

    # ======= تاب التحاليل =======
    with tab4:
        st.info("🔬 ارفع صورة التحليل لاقتراح العلاج.")
        lab_file = st.file_uploader(
            "صورة التحليل",
            type=['jpg', 'png', 'jpeg'],
            key="lab"
        )
        if lab_file and st.button("🧬 قراءة النتيجة"):
            with st.spinner("🔍 جاري تحميل محرك القراءة..."):
                reader = get_ocr_reader()
            with st.spinner("🔍 جاري تحليل النتائج..."):
                img = Image.open(lab_file)
                img_np = np.array(img)
                results = reader.readtext(img_np, detail=0)
                lab_text = " ".join(results)

            with st.spinner("💡 جاري اقتراح الأدوية..."):
                q = (
                    f"هذا نص من نتيجة تحليل: '{lab_text}'. "
                    "استخرج القيم غير الطبيعية، استنتج التشخيص، "
                    "واقترح الأدوية العلمية المناسبة للحالة "
                    "مع نصيحة للمريض باللهجة المصرية."
                )
                ans = ask_smart_assistant(llm, vector_store, q)
                st.markdown(
                    f"<div class='report-card'><h3>🩸 تقرير التحليل والعلاج:</h3>{ans}</div>",
                    unsafe_allow_html=True
                )


if __name__ == "__main__":
    main()
