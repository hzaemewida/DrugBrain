import os
import streamlit as st
import numpy as np
from PIL import Image
import gc

# ====== 1. إعدادات الصفحة ======
st.set_page_config(
    page_title="Drugbrain Intelligence OS", 
    layout="wide", 
    page_icon="🛸"
)

st.markdown("""
    <style>
    @keyframes gradient-shift { 
        0% {background-position:0% 50%} 
        50% {background-position:100% 50%} 
        100% {background-position:0% 50%} 
    }
    .animated-title {
        background: linear-gradient(270deg, #ff007f, #7f00ff, #00d2ff, #3a7bd5);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 5s ease infinite;
        text-align: center; font-size: 3.2rem; font-weight: 900;
    }
    .report-card {
        background: white; padding: 25px; border-radius: 15px; 
        border-right: 10px solid #7f00ff; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.1); 
        color: #1a1a1a; direction: rtl; 
        text-align: right; line-height: 1.6;
    }
    .stButton>button { 
        background: linear-gradient(90deg, #7f00ff, #ff007f); 
        color: white; border-radius: 10px; border:none; 
    }
    </style>
    <h1 class="animated-title">🛸 Drugbrain Intelligence OS 🧬</h1>
""", unsafe_allow_html=True)


# ====== 2. Backend Functions ======
@st.cache_resource(show_spinner="🤖 جاري تحميل الذكاء الاصطناعي...")
def get_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(
        api_key=st.secrets["GROQ_API_KEY"], 
        model_name="llama-3.3-70b-versatile", 
        temperature=0.1,
        # ✅ زيادة timeout يحل مشكلة الانقطاع
        request_timeout=120,
        max_retries=3,
    )

@st.cache_resource(show_spinner="📚 جاري تحميل نموذج اللغة...")
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource(show_spinner="📖 جاري تحميل المراجع الطبية...")
def get_vector_store():
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import CharacterTextSplitter
    
    embed_model = get_embeddings()
    index_path = "faiss_index_v3"
    
    if os.path.exists(index_path):
        try:
            return FAISS.load_local(
                index_path, 
                embed_model, 
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.warning(f"⚠️ فشل تحميل الـ index، هيتعمل من جديد: {e}")
    
    books = ["Clinical Pharmacology Made Incredibly Easy (3rd Ed.).pdf"]
    all_docs = []
    
    for b in books:
        if os.path.exists(b):
            try:
                loader = PyPDFLoader(b)
                all_docs.extend(loader.load())
            except Exception as e:
                st.error(f"❌ خطأ في تحميل {b}: {e}")
    
    if not all_docs:
        return None
    
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs_split = splitter.split_documents(all_docs)
    
    v_store = FAISS.from_documents(docs_split, embed_model)
    v_store.save_local(index_path)
    
    return v_store

@st.cache_resource(show_spinner="👁️ جاري تحميل محرك OCR...")
def get_ocr_reader():
    import easyocr
    return easyocr.Reader(['en', 'ar'], gpu=False, download_enabled=True)


def ask_drugbrain(llm, v_store, query, is_table=False):
    """الدالة الرئيسية للسؤال مع معالجة الأخطاء"""
    try:
        docs = v_store.similarity_search(query, k=3)
        context = "\n".join([d.page_content for d in docs])
        
        table_instr = (
            "\nهام: اعرض الأدوية في جدول Markdown "
            "(الدواء | الجرعة | الاستخدام | ملاحظات)." 
            if is_table else ""
        )
        
        prompt = f"""
أنت خبير صيدلي متخصص. استخدم السياق التالي للإجابة.

السياق من المراجع الطبية:
{context}

السؤال: {query}
{table_instr}

أجب باللهجة المصرية العامية بشكل واضح ومفصل.
إذا المعلومة مش في السياق، قول ذلك بصراحة.
"""
        response = llm.invoke(prompt)
        return response.content
        
    except Exception as e:
        return f"❌ حدث خطأ: {str(e)}\nحاول تاني أو دوس على زرار Reboot."


def process_image_ocr(uploaded_file):
    """معالجة الصورة بشكل آمن"""
    try:
        reader = get_ocr_reader()
        img = Image.open(uploaded_file)
        
        # ✅ تصغير الصورة لو كبيرة (بيقلل وقت المعالجة)
        if img.width > 1500 or img.height > 1500:
            img.thumbnail((1500, 1500), Image.LANCZOS)
        
        img_array = np.array(img)
        result = reader.readtext(img_array, detail=0)
        
        # تنظيف الذاكرة
        del img_array
        gc.collect()
        
        return " ".join(result)
        
    except Exception as e:
        st.error(f"❌ خطأ في قراءة الصورة: {e}")
        return None


# ====== 3. Frontend ======
def main():
    # ✅ نظام إدارة الذاكرة
    if 'session_uses' not in st.session_state: 
        st.session_state.session_uses = 0
    if 'results_cache' not in st.session_state:
        st.session_state.results_cache = {}
        
    st.session_state.session_uses += 1
    if st.session_state.session_uses % 5 == 0: 
        gc.collect()

    # ====== Sidebar ======
    with st.sidebar:
        st.header("⚙️ إدارة النظام")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("♻️ Reboot", use_container_width=True):
                st.cache_resource.clear()
                st.session_state.clear()
                gc.collect()
                st.rerun()
        with col2:
            if st.button("🗑️ مسح", use_container_width=True):
                st.session_state.results_cache = {}
                gc.collect()
                st.success("✅ تم!")
        
        st.divider()
        st.info("💡 لو التطبيق تقل، دوس **Reboot**")
        st.metric("عدد الاستخدامات", st.session_state.session_uses)

    # ====== تحميل النماذج ======
    llm = get_llm()
    v_store = get_vector_store()
    
    if not v_store:
        st.error("⚠️ ملفات PDF مش موجودة في المشروع!")
        st.code("تأكد إن الملف موجود:\nClinical Pharmacology Made Incredibly Easy (3rd Ed.).pdf")
        return

    # ====== Tabs ======
    tab1, tab2, tab3, tab4 = st.tabs([
        "👁️ الروشتات", 
        "💬 استفسار طبي", 
        "⚠️ التعارضات", 
        "🩸 التحاليل"
    ])

    # ------ Tab 1: الروشتات ------
    with tab1:
        st.subheader("👁️ تحليل الروشتات الطبية")
        f1 = st.file_uploader(
            "ارفع صورة الروشتة:", 
            type=['jpg','png','jpeg'], 
            key="rx"
        )
        
        if f1:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(f1, caption="الروشتة المرفوعة", use_container_width=True)
            
            if st.button("🚀 ابدأ التحليل", key="b1", use_container_width=True):
                # ✅ استخدام placeholder لتجنب إعادة الرسم
                status = st.empty()
                result_area = st.empty()
                
                with status.status("جاري العمل...") as s:
                    s.write("👀 جاري قراءة الروشتة بالـ OCR...")
                    raw_text = process_image_ocr(f1)
                    
                    if raw_text:
                        s.write(f"📝 النص المستخرج: `{raw_text[:100]}...`")
                        s.write("🩺 جاري تحليل الأدوية بالذكاء الاصطناعي...")
                        
                        res = ask_drugbrain(
                            llm, v_store, 
                            f"حلل الروشتة دي وطلع الأدوية وجرعاتها: {raw_text}", 
                            is_table=True
                        )
                        s.update(label="✅ تم التحليل!", state="complete")
                        
                        result_area.markdown(
                            f"<div class='report-card'><h3>🩺 التقرير:</h3>{res}</div>", 
                            unsafe_allow_html=True
                        )

    # ------ Tab 2: الاستفسار ------
    with tab2:
        st.subheader("💬 استفسار طبي")
        q = st.text_input(
            "اسأل عن أي دواء أو حالة مرضية:", 
            placeholder="مثال: ما هي أعراض جانبية الأسبرين؟"
        )
        
        if q and st.button("🔍 بحث", key="b2", use_container_width=True):
            with st.spinner("🤔 جاري البحث..."):
                res = ask_drugbrain(llm, v_store, q)
            st.markdown(
                f"<div class='report-card'>{res}</div>", 
                unsafe_allow_html=True
            )

    # ------ Tab 3: التعارضات ------
    with tab3:
        st.subheader("⚠️ فحص التعارضات الدوائية")
        drugs = st.text_area(
            "أدخل الأدوية (كل دواء في سطر):",
            placeholder="أسبرين\nوارفارين\nإيبوبروفين",
            height=120
        )
        
        if drugs and st.button("🚨 فحص التعارضات", key="b3", use_container_width=True):
            with st.spinner("⚠️ جاري الفحص..."):
                res = ask_drugbrain(
                    llm, v_store, 
                    f"هل فيه تعارضات خطيرة بين الأدوية دي؟ {drugs}", 
                    is_table=True
                )
            st.markdown(
                f"<div class='report-card'>{res}</div>", 
                unsafe_allow_html=True
            )

    # ------ Tab 4: التحاليل ------
    with tab4:
        st.subheader("🩸 تحليل نتائج التحاليل")
        f2 = st.file_uploader(
            "ارفع صورة التحليل:", 
            type=['jpg','png','jpeg'], 
            key="lab"
        )
        
        if f2:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(f2, caption="صورة التحليل", use_container_width=True)
            
            if st.button("🧬 تحليل النتائج", key="b4", use_container_width=True):
                status2 = st.empty()
                result_area2 = st.empty()
                
                with status2.status("جاري التحليل...") as s:
                    s.write("🔍 جاري قراءة التحليل...")
                    raw_lab = process_image_ocr(f2)
                    
                    if raw_lab:
                        s.write("🧬 جاري كتابة التقرير...")
                        res = ask_drugbrain(
                            llm, v_store, 
                            f"حلل نتائج التحليل ده وقول إيه المعدل الطبيعي: {raw_lab}"
                        )
                        s.update(label="✅ تم!", state="complete")
                        
                        result_area2.markdown(
                            f"<div class='report-card'><h3>🩸 نتائج التحليل:</h3>{res}</div>", 
                            unsafe_allow_html=True
                        )

if __name__ == "__main__":
    main()
