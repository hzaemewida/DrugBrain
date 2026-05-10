import os
import markdown
import streamlit as st
import numpy as np
from PIL import Image
import gc

# ====== 1. إعدادات الصفحة والتنسيق ======
st.set_page_config(page_title="Drugbrain Intelligence OS", layout="wide", page_icon="🛸")

st.markdown("""
    <style>
    @keyframes gradient-shift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* ===== Floating Particles ===== */
    .particles-container {
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        pointer-events: none;
        z-index: 0;
        overflow: hidden;
    }
    .particle {
        position: absolute;
        border-radius: 50%;
        opacity: 0;
        animation: float-up linear infinite;
    }
    @keyframes float-up {
        0%   { transform: translateY(100vh) scale(0);   opacity: 0; }
        10%  { opacity: 1; }
        90%  { opacity: 0.6; }
        100% { transform: translateY(-10vh) scale(1.2); opacity: 0; }
    }
    .p1  { width:4px;  height:4px;  left:5%;   background:#7f00ff; animation-duration:12s; animation-delay:0s;    }
    .p2  { width:3px;  height:3px;  left:12%;  background:#00d2ff; animation-duration:15s; animation-delay:1.5s;  }
    .p3  { width:5px;  height:5px;  left:20%;  background:#ff007f; animation-duration:10s; animation-delay:3s;    }
    .p4  { width:2px;  height:2px;  left:28%;  background:#7f00ff; animation-duration:18s; animation-delay:0.5s;  }
    .p5  { width:4px;  height:4px;  left:35%;  background:#00d2ff; animation-duration:13s; animation-delay:2s;    }
    .p6  { width:3px;  height:3px;  left:42%;  background:#7f00ff; animation-duration:16s; animation-delay:4s;    }
    .p7  { width:6px;  height:6px;  left:50%;  background:#00d2ff; animation-duration:11s; animation-delay:1s;    }
    .p8  { width:2px;  height:2px;  left:57%;  background:#ff007f; animation-duration:14s; animation-delay:2.5s;  }
    .p9  { width:4px;  height:4px;  left:63%;  background:#7f00ff; animation-duration:17s; animation-delay:0s;    }
    .p10 { width:3px;  height:3px;  left:70%;  background:#00d2ff; animation-duration:12s; animation-delay:3.5s;  }
    .p11 { width:5px;  height:5px;  left:76%;  background:#7f00ff; animation-duration:15s; animation-delay:1s;    }
    .p12 { width:2px;  height:2px;  left:82%;  background:#ff007f; animation-duration:10s; animation-delay:4.5s;  }
    .p13 { width:4px;  height:4px;  left:88%;  background:#00d2ff; animation-duration:13s; animation-delay:2s;    }
    .p14 { width:3px;  height:3px;  left:93%;  background:#7f00ff; animation-duration:16s; animation-delay:0.5s;  }
    .p15 { width:5px;  height:5px;  left:8%;   background:#00d2ff; animation-duration:19s; animation-delay:5s;    }
    .p16 { width:2px;  height:2px;  left:17%;  background:#ff007f; animation-duration:11s; animation-delay:1.5s;  }
    .p17 { width:4px;  height:4px;  left:32%;  background:#7f00ff; animation-duration:14s; animation-delay:3s;    }
    .p18 { width:3px;  height:3px;  left:47%;  background:#00d2ff; animation-duration:17s; animation-delay:2.5s;  }
    .p19 { width:5px;  height:5px;  left:68%;  background:#7f00ff; animation-duration:12s; animation-delay:4s;    }
    .p20 { width:2px;  height:2px;  left:85%;  background:#ff007f; animation-duration:15s; animation-delay:1s;    }
    .particle { box-shadow: 0 0 6px 2px currentColor; }

    /* ===== Glassmorphism Report Card ===== */
    @keyframes card-glow {
        0%   { border-color: rgba(127, 0, 255, 0.4); box-shadow: 0 8px 32px rgba(127,0,255,0.15); }
        50%  { border-color: rgba(0, 210, 255, 0.6); box-shadow: 0 8px 32px rgba(0,210,255,0.2);  }
        100% { border-color: rgba(127, 0, 255, 0.4); box-shadow: 0 8px 32px rgba(127,0,255,0.15); }
    }
    .report-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        padding: 28px;
        border-radius: 20px;
        border: 1px solid rgba(127, 0, 255, 0.4);
        animation: card-glow 4s ease-in-out infinite;
        color: inherit;
        direction: rtl;
        text-align: right;
        line-height: 1.8;
        position: relative;
        z-index: 1;
        margin-top: 12px;
    }
    .report-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 20px 20px 0 0;
        background: linear-gradient(90deg, #7f00ff, #00d2ff, #ff007f);
    }
    .report-card h3 {
        background: linear-gradient(90deg, #7f00ff, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 12px;
        font-size: 1.2rem;
    }

    /* ===== Table Styling ===== */
    .report-card table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 12px;
        direction: rtl;
    }
    .report-card th {
        background: linear-gradient(90deg, rgba(127,0,255,0.3), rgba(0,210,255,0.3));
        color: inherit;
        padding: 10px 14px;
        text-align: right;
        font-weight: 700;
        border-bottom: 2px solid rgba(127,0,255,0.5);
    }
    .report-card td {
        padding: 9px 14px;
        text-align: right;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    .report-card tr:hover td {
        background: rgba(127,0,255,0.08);
    }

    /* ===== Title ===== */
    .animated-title {
        background: linear-gradient(270deg, #ff007f, #7f00ff, #00d2ff, #3a7bd5);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 5s ease infinite;
        text-align: center; font-size: 3.2rem; font-weight: 900;
        position: relative; z-index: 1;
    }
    .stButton>button {
        background: linear-gradient(90deg, #7f00ff, #ff007f);
        color: white; border-radius: 10px; border: none;
    }

    /* ===== Patient Badge ===== */
    .patient-badge {
        background: rgba(127,0,255,0.15);
        border: 1px solid rgba(127,0,255,0.4);
        border-radius: 12px;
        padding: 10px 14px;
        font-size: 0.85rem;
        direction: rtl;
        text-align: right;
        margin-bottom: 8px;
        color: inherit;
    }
    </style>

    <!-- Particles -->
    <div class="particles-container">
      <div class="particle p1"></div><div class="particle p2"></div>
      <div class="particle p3"></div><div class="particle p4"></div>
      <div class="particle p5"></div><div class="particle p6"></div>
      <div class="particle p7"></div><div class="particle p8"></div>
      <div class="particle p9"></div><div class="particle p10"></div>
      <div class="particle p11"></div><div class="particle p12"></div>
      <div class="particle p13"></div><div class="particle p14"></div>
      <div class="particle p15"></div><div class="particle p16"></div>
      <div class="particle p17"></div><div class="particle p18"></div>
      <div class="particle p19"></div><div class="particle p20"></div>
    </div>

    <h1 class="animated-title">🛸 Drugbrain Intelligence OS 🧬</h1>
""", unsafe_allow_html=True)

# ====== 2. الدوال الأساسية (Backend) ======
@st.cache_resource
def get_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(api_key=st.secrets["GROQ_API_KEY"], model_name="llama-3.3-70b-versatile", temperature=0.1)

@st.cache_resource
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_vector_store():
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import CharacterTextSplitter

    embed_model = get_embeddings()
    index_path = "faiss_index_v3"

    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embed_model, allow_dangerous_deserialization=True)

    books = ["Clinical Pharmacology Made Incredibly Easy (3rd Ed.).pdf"]
    all_docs = []
    for b in books:
        if os.path.exists(b):
            loader = PyPDFLoader(b)
            all_docs.extend(loader.load())

    if not all_docs: return None

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    v_store = FAISS.from_documents(splitter.split_documents(all_docs), embed_model)
    v_store.save_local(index_path)
    return v_store

@st.cache_resource
def get_ocr_reader():
    import easyocr
    return easyocr.Reader(['en'], gpu=False, download_enabled=True)

def ask_drugbrain(llm, v_store, query, is_table=False, age=30, weight=70, conditions=[]):
    docs = v_store.similarity_search(query, k=3)
    context = "\n".join([d.page_content for d in docs])

    patient_info = f"المريض عمره {age} سنة، وزنه {weight} كجم."
    if conditions:
        patient_info += f" عنده: {', '.join(conditions)}."

    table_instr = "\nهام: اعرض الأدوية في جدول Markdown (الدواء | الجرعة | ملاحظات)." if is_table else ""

    prompt = f"""بيانات المريض: {patient_info}
السياق الطبي: {context}
السؤال: {query}{table_instr}
أجب باللهجة المصرية العامية كخبير صيدلي، مع مراعاة بيانات المريض في تحديد الجرعات والتحذيرات."""

    result = llm.invoke(prompt).content
    return markdown.markdown(result, extensions=['tables'])

# ====== 3. واجهة التطبيق (Frontend) ======
def main():
    if 'session_uses' not in st.session_state: st.session_state.session_uses = 0
    st.session_state.session_uses += 1
    if st.session_state.session_uses % 5 == 0: gc.collect()

    # ===== Sidebar =====
    with st.sidebar:
        st.header("👤 بيانات المريض")
        age = st.number_input("العمر (سنة)", min_value=1, max_value=120, value=30)
        weight = st.number_input("الوزن (كجم)", min_value=10, max_value=200, value=70)
        conditions = st.multiselect(
            "حالات خاصة",
            ["فشل كلوي", "فشل كبدي", "حمل", "رضاعة", "سكر", "ضغط", "حساسية من البنسلين"]
        )

        st.divider()

        st.header("⚙️ إدارة النظام")
        if st.button("♻️ تنشيط الذاكرة (Reboot)"):
            st.cache_resource.clear()
            gc.collect()
            st.rerun()
        st.info("لو التطبيق تقل معاك، دوس هنا.")

    # إظهار بيانات المريض فوق المحتوى
    conditions_text = f" | {', '.join(conditions)}" if conditions else ""
    st.markdown(f"""
        <div class='patient-badge'>
        👤 <b>المريض:</b> {age} سنة | {weight} كجم{conditions_text}
        </div>
    """, unsafe_allow_html=True)

    llm = get_llm()
    v_store = get_vector_store()

    if not v_store:
        st.error("⚠️ المراجع الطبية (PDF) مش موجودة في الـ GitHub.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["👁️ الروشتات", "💬 استفسار طبي", "⚠️ التعارضات", "🩸 التحاليل"])

    with tab1:
        f1 = st.file_uploader("ارفع صورة الروشتة:", type=['jpg','png','jpeg'], key="rx")
        if f1 and st.button("🚀 ابدأ التحليل", key="b1"):
            with st.spinner("👀 جاري قراءة الروشتة..."):
                reader = get_ocr_reader()
                img = np.array(Image.open(f1))
                raw_text = " ".join(reader.readtext(img, detail=0))
                del img; gc.collect()
            with st.spinner("🩺 جاري استخراج الأدوية..."):
                res = ask_drugbrain(llm, v_store, f"حلل الروشتة دي وطلع الأدوية: {raw_text}",
                                   is_table=True, age=age, weight=weight, conditions=conditions)
                st.markdown(f"<div class='report-card'><h3>🩺 التقرير</h3>{res}</div>", unsafe_allow_html=True)

    with tab2:
        q = st.text_input("اسأل عن أي دواء أو حالة:")
        if q and st.button("🔍 بحث", key="b2"):
            res = ask_drugbrain(llm, v_store, q, age=age, weight=weight, conditions=conditions)
            st.markdown(f"<div class='report-card'>{res}</div>", unsafe_allow_html=True)

    with tab3:
        drugs = st.text_area("أدخل الأدوية لفحص التفاعلات:")
        if drugs and st.button("🚨 فحص", key="b3"):
            res = ask_drugbrain(llm, v_store, f"هل فيه تعارض بين: {drugs}؟",
                               is_table=True, age=age, weight=weight, conditions=conditions)
            st.markdown(f"<div class='report-card'>{res}</div>", unsafe_allow_html=True)

    with tab4:
        f2 = st.file_uploader("ارفع صورة التحليل:", type=['jpg','png','jpeg'], key="lab")
        if f2 and st.button("🧬 تحليل النتائج", key="b4"):
            with st.spinner("🔍 جاري الفحص..."):
                reader = get_ocr_reader()
                raw_lab = " ".join(reader.readtext(np.array(Image.open(f2)), detail=0))
                gc.collect()
            with st.spinner("🩸 جاري كتابة التقرير..."):
                res = ask_drugbrain(llm, v_store, f"حلل التحليل ده: {raw_lab}",
                                   age=age, weight=weight, conditions=conditions)
                st.markdown(f"<div class='report-card'><h3>🩸 نتائج التحليل</h3>{res}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
