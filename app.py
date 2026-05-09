import os
import base64
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# ===== إعداد الصفحة =====
st.set_page_config(
    page_title="Drugbrain Intelligence OS",
    layout="wide",
    page_icon="🛸"
)

st.markdown("""
<h1 style='text-align:center;
background: linear-gradient(90deg,#7f00ff,#00d2ff);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
font-size:3rem;'>
🛸 Drugbrain Intelligence OS 🧬
</h1>
""", unsafe_allow_html=True)

# ===== API KEY =====
def get_api_key():
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    if not api_key:
        st.error("❌ لم يتم العثور على GROQ_API_KEY. أضفه في Secrets.")
        st.stop()
    return api_key

@st.cache_resource
def get_text_model():
    return ChatGroq(
        api_key=get_api_key(),
        model_name="llama-3.3-70b-versatile",
        temperature=0.1
    )

@@st.cache_resource
def get_vision_model():
    return ChatGroq(
        api_key=get_api_key(),
        model_name="llama-3.2-90b-vision-preview",
        temperature=0.1
    )

def ask_text_model(prompt):
    model = get_text_model()
    response = model.invoke(prompt)
    return response.content

def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.getvalue()).decode()

def ask_vision_model(uploaded_file, prompt):
    model = get_vision_model()
    img_base64 = encode_image(uploaded_file)

    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
    ])

    response = model.invoke([message])
    return response.content


# ===== التابات =====
tab1, tab2, tab3, tab4 = st.tabs([
    "👁️ الروشتات",
    "💬 استفسار طبي",
    "⚠️ التعارضات",
    "🩸 التحاليل"
])

# ===== الروشتات =====
with tab1:
    rx_file = st.file_uploader("ارفع صورة الروشتة", type=["jpg", "png", "jpeg"])
    if rx_file and st.button("تحليل الروشتة"):
        with st.spinner("جارى التحليل..."):
            prompt = """
            أنت صيدلي مصري خبير.
            اقرأ الروشتة الطبية وحدد:
            - الأدوية المكتوبة
            - الاستخدام
            - التشخيص المحتمل
            - نصيحة للمريض
            اكتب تقرير منظم باللهجة المصرية.
            """
            result = ask_vision_model(rx_file, prompt)
            st.success(result)

# ===== استفسار طبي =====
with tab2:
    question = st.text_input("اسأل عن أي دواء")
    if question and st.button("بحث"):
        with st.spinner("جارى البحث..."):
            prompt = f"""
            أنت طبيب وصيدلي مصري خبير.
            أجب بدقة وبطريقة مبسطة باللهجة المصرية:
            {question}
            """
            result = ask_text_model(prompt)
            st.success(result)

# ===== التعارضات =====
with tab3:
    drugs = st.text_area("اكتب الأدوية مفصولة بفاصلة")
    if drugs and st.button("فحص التعارضات"):
        with st.spinner("جارى الفحص..."):
            prompt = f"""
            افحص التفاعلات الدوائية بين:
            {drugs}
            هل يوجد تعارض خطير؟
            وما البديل المناسب؟
            اكتب باللهجة المصرية.
            """
            result = ask_text_model(prompt)
            st.success(result)

# ===== التحاليل =====
with tab4:
    lab_file = st.file_uploader("ارفع صورة التحليل", type=["jpg", "png", "jpeg"])
    if lab_file and st.button("تحليل النتيجة"):
        with st.spinner("جارى التحليل..."):
            prompt = """
            أنت طبيب باطني خبير.
            اقرأ نتيجة التحليل وحدد:
            - القيم غير الطبيعية
            - التشخيص المحتمل
            - العلاج المقترح
            اكتب باللهجة المصرية.
            """
            result = ask_vision_model(lab_file, prompt)
            st.success(result)
