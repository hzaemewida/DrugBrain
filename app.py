import os
import streamlit as st
import numpy as np
from PIL import Image
import gc
import requests
import json

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
    .warning-box {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 8px;
        color: #856404;
        direction: rtl;
    }
    </style>
    <h1 class="animated-title">🛸 Drugbrain Intelligence OS 🧬</h1>
""", unsafe_allow_html=True)


# ====== Backend - الدوال المحسّنة ======

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
        max_retries=3           # ← يحاول 3 مرات
    )

@st.cache_resource
def get_embeddings():
    """نموذج خفيف جداً - 50MB بس!"""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource
def get_vector_store():
    """قاعدة البيانات الطبية - محسّنة"""
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    embed_model = get_embeddings()
    index_path = "faiss_index_v3"

    # تحميل Index لو موجود
    if os.path.exists(index_path):
        try:
            return FAISS.load_local(
                index_path,
                embed_model,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.warning(f"⚠️ إعادة بناء الـ Index: {e}")

    # بناء Index من الصفر
    books = ["Clinical Pharmacology Made Incredibly Easy (3rd Ed.).pdf"]
    all_docs = []
    
    for book in books:
        if os.path.exists(book):
            try:
                loader = PyPDFLoader(book)
                all_docs.extend(loader.load())
            except Exception as e:
                st.error(f"⚠️ خطأ في قراءة {book}: {e}")

    if not all_docs:
        return None

    try:
        # Splitter محسّن - يوفر ذاكرة
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,      # ← أصغر من 1000
            chunk_overlap=50,    # ← أقل overlap
            length_function=len
        )
        docs_split = splitter.split_documents(all_docs)
        
        # بناء FAISS Index
        v_store = FAISS.from_documents(docs_split, embed_model)
        v_store.save_local(index_path)
        
        # تنظيف الذاكرة
        del all_docs, docs_split
        gc.collect()
        
        return v_store
    except Exception as e:
        st.error(f"⚠️ خطأ في بناء Index: {e}")
        return None


def ocr_with_api(image_file, language='ara'):
    """
    OCR عبر API خارجي - يوفر 450MB من الذاكرة!
    يدعم العربي والإنجليزي
    """
    try:
        url = "https://api.ocr.space/parse/image"
        
        # تحويل الصورة لـ bytes
        image_bytes = image_file.getvalue()
        
        # إعدادات الـ Request
        files = {'file': image_bytes}
        data = {
            'apikey': st.secrets.get("OCR_API_KEY", "K87899142388957"),  # مفتاح تجريبي
            'language': language,  # ara للعربي, eng للإنجليزي
            'isOverlayRequired': False,
            'detectOrientation': True,
            'scale': True,
            'OCREngine': 2  # محرك أحدث
        }
        
        # إرسال الطلب
        response = requests.post(url, files=files, data=data, timeout=30)
        result = response.json()
        
        # معالجة النتيجة
        if result.get('IsErroredOnProcessing'):
            error_msg = result.get('ErrorMessage', ['خطأ غير معروف'])[0]
            st.error(f"⚠️ خطأ في OCR: {error_msg}")
            return ""
        
        if result.get('ParsedResults'):
            text = result['ParsedResults'][0]['ParsedText']
            return text.strip()
        
        return ""
        
    except requests.exceptions.Timeout:
        st.error("⚠️ انتهى وقت الاتصال بخدمة OCR - حاول مرة أخرى")
        return ""
    except Exception as e:
        st.error(f"⚠️ خطأ في OCR: {str(e)}")
        return ""


def ask_drugbrain(llm, v_store, query, is_table=False):
    """محرك السؤال والجواب - محمي بالكامل"""
    try:
        # البحث في قاعدة البيانات
        docs = v_store.similarity_search(query, k=3)
        context = "\n\n".join([f"📄 مصدر {i+1}:\n{d.page_content}" for i, d in enumerate(docs)])
        
        # تعليمات الجدول
        table_instruction = ""
        if is_table:
            table_instruction = """

⚠️ **مهم جداً**: اعرض النتائج في جدول Markdown بهذا الشكل بالضبط:

| الدواء | الجرعة | التوقيت | ملاحظات مهمة |
|--------|--------|---------|--------------|
| ... | ... | ... | ... |
"""
        
        # البرومبت المحسّن
        prompt = f"""أنت **د. Drug Brain** - صيدلي مصري خبير بتتكلم باللهجة المصرية الفصيحة.

📚 **المعلومات الطبية المتاحة:**
{context}

❓ **سؤال المريض:**
{query}

{table_instruction}

📋 **تعليمات الإجابة:**
1. اتكلم باللهجة المصرية بس بشكل واضح ومفهوم
2. لو المعلومة مش موجودة في السياق، قول "الأفضل تستشير دكتورك"
3. اذكر أي تحذيرات أو آثار جانبية مهمة
4. لو فيه جرعات، اذكرها بوضوح
5. خلي الإجابة مختصرة ومفيدة (مش أكتر من 200 كلمة)
6. استخدم إيموجي مناسب عشان توضح النقاط المهمة

💊 **إجابتك:**
"""
        
        # استدعاء الـ LLM
        response = llm.invoke(prompt).content
        return response
        
    except Exception as e:
        error_msg = f"""
        <div class='error-box'>
        ⚠️ <strong>حصلت مشكلة تقنية:</strong><br>
        <code>{str(e)}</code><br><br>
        💡 <strong>الحلول المقترحة:</strong><br>
        • اضغط "♻️ تنشيط الذاكرة" في الشريط الجانبي<br>
        • حاول مرة تانية بعد 10 ثواني<br>
        • لو المشكلة مستمرة، جرب صياغة السؤال بطريقة تانية
        </div>
        """
        return error_msg


# ====== Frontend - الواجهة ======

def main():
    # نظام التنظيف التلقائي
    if 'session_uses' not in st.session_state:
        st.session_state.session_uses = 0
        st.session_state.total_images = 0
        
    st.session_state.session_uses += 1
    
    # تنظيف كل 3 استخدامات (بدل 5)
    if st.session_state.session_uses % 3 == 0:
        gc.collect()

    # ====== Sidebar ======
    with st.sidebar:
        st.header("⚙️ لوحة التحكم")
        
        # زر التنشيط
        if st.button("♻️ تنشيط الذاكرة", use_container_width=True):
            st.cache_resource.clear()
            gc.collect()
            st.success("✅ تم التنشيط بنجاح!")
            st.balloons()
            st.rerun()
        
        st.info("💡 لو التطبيق بطّأ، دوس هنا")
        
        st.divider()
        
        # إحصائيات
        col1, col2 = st.columns(2)
        col1.metric("الاستخدامات", st.session_state.session_uses)
        col2.metric("الصور", st.session_state.total_images)
        
        st.caption("🔄 تنظيف تلقائي كل 3 استخدامات")
        
        st.divider()
        
        # معلومات النظام
        with st.expander("ℹ️ عن النظام"):
            st.markdown("""
            **Drugbrain Intelligence OS v2.0**
            
            🧠 **التقنيات:**
            - LLM: Groq Llama 3.3 70B
            - OCR: OCR.space API
            - Vector DB: FAISS
            - Embeddings: MiniLM-L3
            
            💾 **استهلاك الذاكرة:**
            - ~700 MB فقط
            - متوافق مع Free Tier
            
            ⚠️ **تنبيه طبي:**
            هذا النظام للاسترشاد فقط
            استشر طبيبك دائماً
            """)

    # ====== تحميل النماذج ======
    llm = get_llm()
    v_store = get_vector_store()

    if not v_store:
        st.error("""
        ⚠️ **قاعدة البيانات الطبية غير متاحة**
        
        **الحل:**
        1. تأكد من وجود ملف PDF في المجلد الرئيسي
        2. الاسم: `Clinical Pharmacology Made Incredibly Easy (3rd Ed.).pdf`
        3. أعد تشغيل التطبيق
        """)
        return

    # ====== التابات الرئيسية ======
    tab1, tab2, tab3, tab4 = st.tabs([
        "👁️ تحليل الروشتات",
        "💬 استفسار طبي",
        "⚠️ فحص التعارضات",
        "🩸 قراءة التحاليل"
    ])

    # ====== TAB 1: الروشتات ======
    with tab1:
        st.markdown("### 📋 رفع صورة الروشتة الطبية")
        st.markdown('<div class="warning-box">📸 ارفع صورة واضحة - يفضل خلفية بيضاء وإضاءة جيدة</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            f1 = st.file_uploader(
                "اختر صورة الروشتة",
                type=['jpg', 'png', 'jpeg'],
                key="rx",
                help="JPG, PNG أو JPEG - حجم أقصى 5MB"
            )
            
            if f1:
                st.image(f1, caption="✅ تم رفع الصورة", use_container_width=True)
        
        with col2:
            if f1:
                lang = st.radio(
                    "لغة الروشتة:",
                    options=['ara', 'eng'],
                    format_func=lambda x: "🇪🇬 عربي" if x == 'ara' else "🇬🇧 إنجليزي",
                    horizontal=True
                )
                
                if st.button("🚀 ابدأ التحليل الآن", key="b1", use_container_width=True, type="primary"):
                    st.session_state.total_images += 1
                    
                    # مرحلة 1: OCR
                    with st.spinner("👀 جاري قراءة النص من الصورة..."):
                        raw_text = ocr_with_api(f1, language=lang)
                        
                        if not raw_text.strip():
                            st.warning("⚠️ لم يتم العثور على نص واضح - جرب:")
                            st.markdown("""
                            - صورة بإضاءة أفضل
                            - زاوية تصوير مباشرة
                            - دقة أعلى
                            """)
                            st.stop()
                        
                        # عرض النص المستخرج
                        with st.expander("📝 النص المستخرج من الصورة"):
                            st.text_area("", raw_text, height=150, disabled=True)
                    
                    # مرحلة 2: التحليل
                    with st.spinner("🩺 جاري تحليل الروشتة وإعداد التقرير الطبي..."):
                        analysis = ask_drugbrain(
                            llm, v_store,
                            f"""حلل الروشتة الطبية التالية بالتفصيل:

{raw_text}

اذكر:
1. جميع الأدوية الموجودة
2. الجرعات المحددة
3. مواعيد الاستخدام
4. أي تحذيرات أو تعارضات محتملة
5. نصائح عامة للمريض""",
                            is_table=True
                        )
                        
                        st.markdown(
                            f"<div class='report-card'><h3>🩺 التقرير الطبي الشامل:</h3>{analysis}</div>",
                            unsafe_allow_html=True
                        )
                        
                        # زر التحميل
                        st.download_button(
                            "📥 تحميل التقرير",
                            data=analysis,
                            file_name="drugbrain_report.txt",
                            mime="text/plain"
                        )
                        
                        gc.collect()

    # ====== TAB 2: استفسار طبي ======
    with tab2:
        st.markdown("### 💊 اسأل عن أي دواء أو حالة طبية")
        
        # أمثلة جاهزة
        examples = st.selectbox(
            "أو اختر من الأمثلة:",
            [
                "اكتب سؤالك...",
                "إيه الفرق بين البروفين والبارامول؟",
                "جرعة الأسبرين المناسبة لكبار السن؟",
                "هل الأوجمنتين آمن للحامل؟",
                "تعارضات دواء الضغط مع المسكنات؟"
            ]
        )
        
        q = st.text_input(
            "سؤالك:",
            value="" if examples == "اكتب سؤالك..." else examples,
            placeholder="مثال: إيه أفضل مسكن لالتهاب المفاصل؟"
        )
        
        if q and q != "اكتب سؤالك..." and st.button("🔍 ابحث الآن", key="b2", use_container_width=True, type="primary"):
            with st.spinner("🔍 جاري البحث في المراجع الطبية..."):
                answer = ask_drugbrain(llm, v_store, q)
                st.markdown(
                    f"<div class='report-card'>{answer}</div>",
                    unsafe_allow_html=True
                )
                gc.collect()

    # ====== TAB 3: التعارضات ======
    with tab3:
        st.markdown("### ⚠️ فحص التفاعلات الدوائية الخطيرة")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            drugs = st.text_area(
                "أدخل الأدوية (كل دواء في سطر):",
                placeholder="مثال:\nبانادول\nبروفين\nأسبرين\nكونكور",
                height=200,
                help="اكتب اسم كل دواء في سطر منفصل"
            )
        
        with col2:
            st.info("""
            **ملاحظات:**
            
            ✅ اكتب الاسم التجاري أو العلمي
            
            ✅ يمكن كتابة بالعربي أو الإنجليزي
            
            ⚠️ النتائج استرشادية فقط
            """)
        
        if drugs and st.button("🚨 فحص التعارضات", key="b3", use_container_width=True, type="primary"):
            with st.spinner("🔍 جاري فحص التفاعلات الدوائية..."):
                interactions = ask_drugbrain(
                    llm, v_store,
                    f"""فحص شامل للتفاعلات الدوائية:

الأدوية المستخدمة:
{drugs}

المطلوب:
1. هل توجد تعارضات خطيرة؟
2. ما هي التفاعلات المحتملة؟
3. ما البدائل الآمنة إن وجدت؟
4. نصائح للاستخدام الآمن

اعرض النتائج في جدول واضح.""",
                    is_table=True
                )
                
                st.markdown(
                    f"<div class='report-card'>{interactions}</div>",
                    unsafe_allow_html=True
                )
                gc.collect()

    # ====== TAB 4: التحاليل ======
    with tab4:
        st.markdown("### 🩸 رفع صورة التحليل الطبي")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            f2 = st.file_uploader(
                "اختر صورة التحليل",
                type=['jpg', 'png', 'jpeg'],
                key="lab",
                help="صورة واضحة لنتيجة التحليل"
            )
            
            if f2:
                st.image(f2, caption="✅ تم رفع التحليل", use_container_width=True)
        
        with col2:
            if f2:
                if st.button("🧬 تحليل النتائج", key="b4", use_container_width=True, type="primary"):
                    st.session_state.total_images += 1
                    
                    with st.spinner("🔍 جاري قراءة نتائج التحليل..."):
                        raw_lab = ocr_with_api(f2, language='eng')
                        
                        if not raw_lab.strip():
                            st.warning("⚠️ لم يتم قراءة النص - جرب صورة أوضح")
                            st.stop()
                        
                        with st.expander("📊 البيانات المستخرجة"):
                            st.text_area("", raw_lab, height=150, disabled=True)
                    
                    with st.spinner("🩸 جاري إعداد التقرير الطبي..."):
                        report = ask_drugbrain(
                            llm, v_store,
                            f"""حلل نتيجة التحليل الطبي التالي:

{raw_lab}

المطلوب:
1. تحديد نوع التحليل
2. القيم الموجودة ومقارنتها بالمعدل الطبيعي
3. أي قيم غير طبيعية (مرتفعة أو منخفضة)
4. التفسير الطبي المبسط
5. متى يجب مراجعة الطبيب
6. نصائح عامة

اعرض النتائج بشكل واضح ومنظم."""
                        )
                        
                        st.markdown(
                            f"<div class='report-card'><h3>🩸 تقرير التحليل:</h3>{report}</div>",
                            unsafe_allow_html=True
                        )
                        gc.collect()

    # ====== Footer ======
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
    ⚠️ <strong>تنبيه طبي مهم:</strong> هذا النظام للاسترشاد فقط ولا يغني عن استشارة الطبيب المختص<br>
    🛸 Powered by <strong>Drugbrain Intelligence OS</strong> | Made with ❤️ in Egypt
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
