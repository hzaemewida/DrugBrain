import os
import streamlit as st
import numpy as np
from PIL import Image
import gc

# ====== إعدادات الصفحة ======
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
        text-align: center;
        font-size: 3.2rem;
        font-weight: 900;
    }
    .report-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        border-right: 10px solid #7f00ff;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        color: #1a1a1a;
        direction: rtl;
        text-align: right;
        line-height: 1.8;
    }
    .stButton>button {
        background: linear-gradient(90deg, #7f00ff, #ff007f);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .error-box {
        background: #ffe6e6;
        border-left: 5px solid #ff0000;
        padding: 15px;
        border-radius: 8px;
        color: #cc0000;
        direction: rtl;
    }
    </style>
    <h1 class="animated-title">🛸 Drugbrain Intelligence OS 🧬</h1>
""", unsafe_allow_html=True)


# ====== Backend - الدوال الأساسية ======
@st.cache_resource
def get_llm():
    """نموذج الذكاء الاصطناعي - محسّن ضد الـ timeout"""
    from langchain_groq import ChatGroq
    return ChatGroq(
        api_key=st.secrets["GROQ_API_KEY"],
        model_name="llama-3.3-70b-versatile",
        temperature=0.1,
        streaming=False,        # ← مفتاح الاستقرار
        request_timeout=120,    # ← 2 دقيقة timeout
        max_retries=3           # ← يحاول 3 مرات لو فشل
    )

@st.cache_resource
def get_embeddings():
    """نموذج تحويل النصوص لأرقام"""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def get_vector_store():
    """قاعدة البيانات الطبية"""
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import CharacterTextSplitter

    embed_model = get_embeddings()
    index_path = "faiss_index_v3"

    # لو الـ index موجود، حمّله مباشرة
    if os.path.exists(index_path):
        try:
            return FAISS.load_local(
                index_path,
                embed_model,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.warning(f"⚠️ مشكلة في تحميل الـ index: {e}")

    # لو مش موجود، اعمله من الـ PDF
    books = ["Clinical Pharmacology Made Incredibly Easy (3rd Ed.).pdf"]
    all_docs = []
    
    for book in books:
        if os.path.exists(book):
            try:
                loader = PyPDFLoader(book)
                all_docs.extend(loader.load())
            except Exception as e:
                st.error(f"⚠️ مشكلة في قراءة {book}: {e}")

    if not all_docs:
        return None

    try:
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs_split = splitter.split_documents(all_docs)
        v_store = FAISS.from_documents(docs_split, embed_model)
        v_store.save_local(index_path)
        return v_store
    except Exception as e:
        st.error(f"⚠️ مشكلة في إنشاء قاعدة البيانات: {e}")
        return None

@st.cache_resource
def get_ocr_reader():
    """محرك قراءة الصور - يدعم عربي وإنجليزي"""
    import easyocr
    try:
        return easyocr.Reader(['en', 'ar'], gpu=False, download_enabled=True)
    except Exception as e:
        st.error(f"⚠️ مشكلة في تحميل EasyOCR: {e}")
        return None

def ask_drugbrain(llm, v_store, query, is_table=False):
    """الدالة الرئيسية للسؤال والجواب - محمية ضد الأخطاء"""
    try:
        # البحث في قاعدة البيانات
        docs = v_store.similarity_search(query, k=3)
        context = "\n".join([d.page_content for d in docs[:3]])
        
        # تعليمات الجدول
        table_instr = "\n\n⚠️ هام: اعرض الأدوية في جدول Markdown بالشكل ده:\n| الدواء | الجرعة | ملاحظات |\n|-------|--------|----------|\n" if is_table else ""
        
        # البرومبت النهائي
        prompt = f"""أنت صيدلي خبير بتتكلم باللهجة المصرية العامية.

📚 السياق الطبي:
{context}

❓ سؤال المريض:
{query}

{table_instr}

📝 تعليمات الإجابة:
- اتكلم باللهجة المصرية بس بشكل مفهوم
- لو مش متأكد، قول "الأفضل تستشير دكتور"
- اذكر أي تحذيرات مهمة
- خلي الإجابة واضحة ومختصرة
"""
        
        response = llm.invoke(prompt).content
        return response
        
    except Exception as e:
        error_msg = f"""
        <div class='error-box'>
        ⚠️ <strong>حصلت مشكلة تقنية:</strong><br>
        {str(e)}<br><br>
        💡 <strong>جرب الحلول دي:</strong><br>
        • اضغط على زرار "تنشيط الذاكرة" في الجانب<br>
        • حاول تاني بعد 10 ثواني<br>
        • لو المشكلة مستمرة، اتواصل مع الدعم الفني
        </div>
        """
        return error_msg


# ====== Frontend - الواجهة الرئيسية ======
def main():
    # نظام التنظيف التلقائي للذاكرة
    if 'session_uses' not in st.session_state:
        st.session_state.session_uses = 0
    st.session_state.session_uses += 1
    
    if st.session_state.session_uses % 5 == 0:
        gc.collect()

    # الشريط الجانبي
    with st.sidebar:
        st.header("⚙️ إدارة النظام")
        
        if st.button("♻️ تنشيط الذاكرة (Reboot)", use_container_width=True):
            st.cache_resource.clear()
            gc.collect()
            st.success("✅ تم التنشيط!")
            st.rerun()
        
        st.info("💡 لو التطبيق بطّأ أو حصل خطأ، دوس هنا.")
        
        st.divider()
        
        st.metric("عدد الاستخدامات", st.session_state.session_uses)
        st.caption("🔄 التنظيف التلقائي كل 5 استخدامات")

    # تحميل النماذج
    llm = get_llm()
    v_store = get_vector_store()

    if not v_store:
        st.error("""
        ⚠️ **المراجع الطبية (PDF) غير موجودة**
        
        الحل:
        1. تأكد إن ملف الـ PDF موجود في المجلد الرئيسي
        2. اسمه: `Clinical Pharmacology Made Incredibly Easy (3rd Ed.).pdf`
        3. لو مش موجود، ارفعه على GitHub
        """)
        return

    # التابات الرئيسية
    tab1, tab2, tab3, tab4 = st.tabs([
        "👁️ تحليل الروشتات",
        "💬 استفسار طبي",
        "⚠️ فحص التعارضات",
        "🩸 قراءة التحاليل"
    ])

    # ====== TAB 1: تحليل الروشتات ======
    with tab1:
        st.subheader("📋 رفع صورة الروشتة الطبية")
        
        f1 = st.file_uploader(
            "اختار صورة الروشتة (JPG, PNG)",
            type=['jpg', 'png', 'jpeg'],
            key="rx",
            help="صورة واضحة للروشتة - يفضل خلفية بيضاء"
        )
        
        if f1:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(f1, caption="الصورة المرفوعة", use_container_width=True)
            
            with col2:
                if st.button("🚀 ابدأ التحليل", key="b1", use_container_width=True):
                    reader = get_ocr_reader()
                    
                    if not reader:
                        st.error("⚠️ محرك OCR مش شغال، جرب تعمل Reboot")
                        return
                    
                    with st.spinner("👀 جاري قراءة النص من الصورة..."):
                        try:
                            img = np.array(Image.open(f1))
                            raw_text = " ".join(reader.readtext(img, detail=0))
                            del img
                            gc.collect()
                            
                            if not raw_text.strip():
                                st.warning("⚠️ مفيش نص واضح في الصورة - جرب صورة أوضح")
                                return
                            
                        except Exception as e:
                            st.error(f"⚠️ مشكلة في قراءة الصورة: {e}")
                            return
                    
                    with st.spinner("🩺 جاري تحليل الأدوية وإعداد التقرير..."):
                        res = ask_drugbrain(
                            llm, v_store,
                            f"حلل الروشتة الطبية دي واستخرج كل الأدوية مع جرعاتها وتعليمات الاستخدام:\n\n{raw_text}",
                            is_table=True
                        )
                        st.markdown(
                            f"<div class='report-card'><h3>🩺 التقرير الطبي:</h3>{res}</div>",
                            unsafe_allow_html=True
                        )

    # ====== TAB 2: استفسار طبي ======
    with tab2:
        st.subheader("💊 اسأل عن أي دواء أو حالة طبية")
        
        q = st.text_input(
            "اكتب سؤالك:",
            placeholder="مثال: إيه الفرق بين البروفين والبارامول؟",
            help="اسأل عن أي دواء أو تفاعلات أو جرعات"
        )
        
        if q and st.button("🔍 ابحث الآن", key="b2", use_container_width=True):
            with st.spinner("🔍 جاري البحث في المراجع الطبية..."):
                res = ask_drugbrain(llm, v_store, q)
                st.markdown(
                    f"<div class='report-card'>{res}</div>",
                    unsafe_allow_html=True
                )

    # ====== TAB 3: فحص التعارضات ======
    with tab3:
        st.subheader("⚠️ فحص التفاعلات الدوائية")
        
        drugs = st.text_area(
            "اكتب أسماء الأدوية (كل واحد في سطر):",
            placeholder="بانادول\nبروفين\nأسبرين",
            height=150,
            help="اكتب الأدوية اللي بتاخدها عشان نشوف لو فيه تعارض"
        )
        
        if drugs and st.button("🚨 فحص التعارضات", key="b3", use_container_width=True):
            with st.spinner("🔍 جاري فحص التفاعلات الدوائية..."):
                res = ask_drugbrain(
                    llm, v_store,
                    f"هل فيه أي تعارض أو تفاعل خطير بين الأدوية دي:\n{drugs}\n\nوضّح التعارضات لو موجودة والبدائل الآمنة.",
                    is_table=True
                )
                st.markdown(
                    f"<div class='report-card'>{res}</div>",
                    unsafe_allow_html=True
                )

    # ====== TAB 4: قراءة التحاليل ======
    with tab4:
        st.subheader("🩸 رفع صورة التحليل الطبي")
        
        f2 = st.file_uploader(
            "اختار صورة التحليل",
            type=['jpg', 'png', 'jpeg'],
            key="lab",
            help="صورة واضحة لنتيجة التحليل"
        )
        
        if f2:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(f2, caption="التحليل المرفوع", use_container_width=True)
            
            with col2:
                if st.button("🧬 تحليل النتائج", key="b4", use_container_width=True):
                    reader = get_ocr_reader()
                    
                    if not reader:
                        st.error("⚠️ محرك OCR مش شغال")
                        return
                    
                    with st.spinner("🔍 جاري قراءة النتائج..."):
                        try:
                            img_arr = np.array(Image.open(f2))
                            raw_lab = " ".join(reader.readtext(img_arr, detail=0))
                            del img_arr
                            gc.collect()
                            
                            if not raw_lab.strip():
                                st.warning("⚠️ مفيش نص واضح - جرب صورة أفضل")
                                return
                                
                        except Exception as e:
                            st.error(f"⚠️ مشكلة في قراءة الصورة: {e}")
                            return
                    
                    with st.spinner("🩸 جاري كتابة التقرير الطبي..."):
                        res = ask_drugbrain(
                            llm, v_store,
                            f"حلل نتيجة التحليل الطبي ده ووضّح إيه القيم الطبيعية وإيه اللي محتاج متابعة:\n\n{raw_lab}"
                        )
                        st.markdown(
                            f"<div class='report-card'><h3>🩸 تقرير التحليل:</h3>{res}</div>",
                            unsafe_allow_html=True
                        )

    # Footer
    st.divider()
    st.caption("⚠️ تنبيه: هذا النظام للاسترشاد فقط - استشر طبيبك دائماً قبل أي قرار طبي")


if __name__ == "__main__":
    main()
