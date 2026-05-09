import os
import base64
import streamlit as st
from PIL import Image

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

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


# ====== جلب API Key ======
def get_api_key():
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    if not api_key:
        st.error("❌ لم يتم العثور على GROQ_API_KEY. أضفه في إعدادات Secrets.")
        st.stop()
    return api_key

# ====== تحميل النموذج اللغوي (للنصوص) ======
@st.cache_resource
def get_llm():
    return ChatGroq(
        api_key=get_api_key(),
        model_name="llama-3.3-70b-versatile",
        temperature=0.1
    )

# ====== تحميل نموذج الـ Embeddings ======
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ====== تحميل قاعدة البيانات الجاهزة ======
@st.cache_resource
def get_vector_store():
    embed_model = get_embeddings()
    # تحميل القاعدة الجاهزة اللي رفعناها على جيت هاب
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embed_model, allow_dangerous_deserialization=True)
    else:
        return None


# ====== دوال المساعدة ======
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

def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

# دالة تحليل الصور باستخدام Groq Vision (بديل قوي لـ EasyOCR وبدون استهلاك رام)
def analyze_image_with_vision(image_base64, prompt):
    chat = ChatGroq(
        api_key=get_api_key(), 
        model_name="llama-3.2-11b-vision-preview", # موديل الرؤية المجاني من Groq
        temperature=0.1
    )
    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
    ])
    response = chat.invoke([msg])
    return response.content


# ====== التطبيق الرئيسي ======
def main():
    llm = get_llm()

    with st.spinner("🧠 جاري تحميل القاعدة الطبية..."):
        vector_store = get_vector_store()

    if not vector_store:
        st.error("❌ لم يتم العثور على مجلد 'faiss_index'. تأكد من رفعه على GitHub.")
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
            with st.spinner("👀 جاري قراءة وتحليل الروشتة بالذكاء الاصطناعي..."):
                img_base64 = encode_image(up_file)
                prompt = (
                    "أنت صيدلي خبير. اقرأ هذه الروشتة الطبية جيداً. "
                    "استنتج الأدوية المكتوبة، استخداماتها، والتشخيص المحتمل، "
                    "واقترح تحاليل طبية ضرورية لو الحالة تستدعي. "
                    "اكتب تقرير منظم باللهجة المصرية."
                )
                ans = analyze_image_with_vision(img_base64, prompt)
                st.markdown(
                    f"<div class='report-card'><h3>🩺 تقرير الروشتة:</h3>{ans}</div>",
                    unsafe_allow_html=True
                )

    # ======= تاب الاستفسار الطبي =======
    with tab2:
        q = st.text_input("اسأل عن أي دواء:")
        if q and st.button("🔍 بحث"):
            with st.spinner("📚 جاري البحث في المراجع..."):
                ans = ask_smart_assistant(llm, vector_store, f"أجب باللهجة المصرية: {q}")
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
                    "هل يوجد تعارض خطير؟ وما البديل؟ الرد باللهجة المصرية."
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
            with st.spinner("🔍 جاري قراءة وتحليل النتائج بالذكاء الاصطناعي..."):
                img_base64 = encode_image(lab_file)
                prompt = (
                    "أنت طبيب باطني خبير. اقرأ نتيجة التحليل الطبي في هذه الصورة. "
                    "استخرج القيم غير الطبيعية، استنتج التشخيص، "
                    "واقترح الأدوية العلمية المناسبة للحالة مع نصيحة للمريض باللهجة المصرية."
                )
                ans = analyze_image_with_vision(img_base64, prompt)
                st.markdown(
                    f"<div class='report-card'><h3>🩸 تقرير التحليل والعلاج:</h3>{ans}</div>",
                    unsafe_allow_html=True
                )


if __name__ == "__main__":
    main()
