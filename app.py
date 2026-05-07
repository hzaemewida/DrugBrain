import streamlit as st
import easyocr
import cv2
from PIL import Image
import numpy as np
import os
from langchain_groq import ChatGroq

st.set_page_config(page_title="Drugbrain", layout="wide", page_icon="🛸")

st.markdown("""
    <style>
    .animated-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(270deg, #ff007f, #7f00ff, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 30px;
    }
    .report-card {
        background: linear-gradient(135deg, #ffffff 0%, #f3f4f6 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #7f00ff;
        margin: 10px 0;
        direction: rtl;
        text-align: right;
    }
    .ocr-card {
        background: #1a1a2e;
        color: #00d2ff;
        padding: 15px;
        border-radius: 10px;
        font-family: monospace;
        margin: 10px 0;
    }
    </style>
    <h1 class="animated-title">🛸 Drugbrain Intelligence OS 🧬</h1>
""", unsafe_allow_html=True)


@st.cache_resource
def load_ocr():
    return easyocr.Reader(["en", "ar"], gpu=False)


@st.cache_resource
def load_llm():
    return ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.1
    )


tab1, tab2, tab3 = st.tabs(["👁️ قراءة الروشتة", "💬 استفسار طبي", "⚠️ التعارضات"])

with tab1:
    st.markdown("### 📋 نظام قراءة الروشتات")
    st.markdown("""
    <div style="background: #fff3cd; border-left: 5px solid #ffc107; padding: 15px; border-radius: 8px;">
    ✨ <b>النظام ده بيقرأ أي روشتة:</b><br>
    ✅ مكتوبة بالإيد<br>
    ✅ صور من الموبايل<br>
    ✅ ورق باهت أو مجعد
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("💡 ارفع الروشتة")
        up_file = st.file_uploader(
            "صورة الروشتة",
            type=["jpg", "png", "jpeg", "webp", "bmp"],
            key="rx"
        )
        if up_file:
            img = Image.open(up_file)
            st.image(img, caption="الصورة الأصلية")
    
    with col2:
        if up_file:
            st.info("⚙️ الإعدادات")
            show_raw = st.checkbox("🔍 عرض النص الخام")
    
    if up_file and st.button("🚀 تحليل الروشتة", type="primary"):
        img = Image.open(up_file)
        img_np = np.array(img)
        
        with st.expander("📷 معالجة الصورة", expanded=True):
            from PIL import ImageEnhance
            pil_img = Image.open(up_file)
            enhanced = ImageEnhance.Sharpness(pil_img).enhance(2.5)
            enhanced = ImageEnhance.Contrast(enhanced).enhance(2.0)
            
            c1, c2 = st.columns(2)
            with c1:
                st.image(img, caption="الأصلية")
            with c2:
                st.image(enhanced, caption="بعد التحسين")
            st.success("✅ تم تحسين الصورة")
        
        with st.expander("📝 استخراج النص", expanded=True):
            with st.spinner("🔍 جاري قراءة الروشتة..."):
                reader = load_ocr()
                results = reader.readtext(img_np, detail=1)
                
                if results:
                    text = " ".join([r[1] for r in results])
                    
                    if show_raw:
                        st.markdown(
                            f"<div class='ocr-card'>{text}</div>",
                            unsafe_allow_html=True
                        )
                    
                    st.success(f"✅ تم استخراج {len(text.split())} كلمة")
                else:
                    st.error("⚠️ لم يتم استخراج نص")
                    st.stop()
        
        with st.expander("🧠 التحليل الطبي", expanded=True):
            with st.spinner("🤖 جاري التحليل بالذكاء الاصطناعي..."):
                llm = load_llm()
                
                prompt = f"""أنت دكتور صيدلي مصري خبير في قراءة الروشتات.

النص المستخرج من الروشتة:
{text}

المطلوب منك:
1. 💊 استخراج أسماء الأدوية الصحيحة
2. 📋 جرعة كل دواء وطريقة الاستخدام
3. 🎯 التشخيص المحتمل للمريض
4. ⚠️ تحذيرات وتنبيهات مهمة
5. 🔬 تحاليل مقترحة
6. 💡 نصائح للمريض

اكتب التقرير بشكل منظم باللهجة المصرية."""
                
                response = llm.invoke(prompt)
                answer = response.content
                
                st.markdown(
                    f"<div class='report-card'><h3>🩺 التقرير الطبي:</h3>{answer}</div>",
                    unsafe_allow_html=True
                )
                
                st.download_button(
                    "📥 تحميل التقرير",
                    data=answer,
                    file_name="report.txt",
                    mime="text/plain"
                )


with tab2:
    st.markdown("### 💬 استفسار طبي")
    q = st.text_area("اسأل عن أي دواء أو حالة طبية:", height=100)
    
    if q and st.button("🔍 بحث", key="search"):
        with st.spinner("📚 جاري البحث..."):
            llm = load_llm()
            
            prompt = f"""أنت دكتور صيدلي مصري خبير.

السؤال: {q}

أجب بشكل واضح ومفصل باللهجة المصرية."""
            
            response = llm.invoke(prompt)
            answer = response.content
            
            st.markdown(
                f"<div class='report-card'>🤖 <b>الإجابة:</b><br>{answer}</div>",
                unsafe_allow_html=True
            )


with tab3:
    st.markdown("### ⚠️ فحص التعارضات الدوائية")
    st.warning("⚡ أدخل أسماء الأدوية لمعرفة التفاعلات الخطيرة")
    
    drugs = st.text_area("أدخل أسماء الأدوية (مثال: Aspirin, Warfarin):", height=100)
    
    if drugs and st.button("🧪 فحص التعارضات", key="check"):
        with st.spinner("🚨 جاري فحص التفاعلات..."):
            llm = load_llm()
            
            prompt = f"""أنت دكتور صيدلي مصري خبير.

الأدوية: {drugs}

المطلوب:
1. ⚠️ هل يوجد تعارض خطير بين هذه الأدوية؟
2. 💊 ما المخاطر المحتملة؟
3. 🔄 ما البدائل الآمنة؟
4. 📋 ما النصائح المهمة؟

اكتب باللهجة المصرية."""
            
            response = llm.invoke(prompt)
            answer = response.content
            
            st.markdown(
                f"<div class='report-card'><h3>⚠️ تقرير التفاعلات:</h3>{answer}</div>",
                unsafe_allow_html=True
            )
