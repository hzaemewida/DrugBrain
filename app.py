import os
import streamlit as st
import numpy as np
from PIL import Image
import gc

# ====== 1. إعدادات الصفحة والتنسيق ======
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
        text-align: center; font-size: 3.2rem; font-weight: 900;
    }
    .report-card {
        background: white; padding: 25px; border-radius: 15px; 
        border-right: 10px solid #7f00ff; box-shadow: 0 4px 15px rgba(0,0,0,0.1); 
        color: #1a1a1a; direction: rtl; text-align: right; line-height: 1.6;
    }
    .stButton>button { background: linear-gradient(90deg, #7f00ff, #ff007f); color: white; border-radius: 10px; border:none; }
    </style>
    <h1 class="animated-title">🛸 Drugbrain Intelligence OS 🧬</h1>
""", unsafe_allow_html=True)

# ====== Pharma Matrix Effect ======
st.markdown("""
<canvas id="pharma-matrix" style="position:fixed;top:0;left:0;z-index:0;pointer-events:none;"></canvas>

<script>
(function() {
  const canvas = document.getElementById('pharma-matrix');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;

  window.addEventListener('resize', () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    init();
  });

  const pharmaWords = [
    'Amoxicillin','Metformin','Lisinopril','Atorvastatin','Omeprazole',
    'Paracetamol','Ibuprofen','Warfarin','Aspirin','Diazepam',
    'Morphine','Insulin','Penicillin','Codeine','Furosemide',
    'Amlodipine','Ciprofloxacin','Azithromycin','Doxycycline','Tramadol',
    'C\u2088H\u2089NO\u2082','C\u2081\u2080H\u2081\u2083N\u2085O\u2084','NaCl','C\u2082H\u2085OH','H\u2082O\u2082',
    'C\u2081\u2086H\u2081\u2089N\u2083O\u2085S','CH\u2083COOH','C\u2089H\u2088O\u2084','C\u2081\u2087H\u2081\u2089NO\u2083','NH\u2083',
    'C\u2086H\u2081\u2082O\u2086','HCl','NaHCO\u2083','KCl',
    'Dopamine','Serotonin','Cortisol','Adrenaline','Melatonin',
    'DNA','RNA','ATP','ADP','GABA'
  ];

  const fontSize = 13;
  const colorThemes = [
    {r:0,   g:255, b:150},
    {r:127, g:0,   b:255},
    {r:0,   g:210, b:255},
    {r:255, g:0,   b:127},
  ];

  let columns, drops, dropWords, dropSpeeds, dropOpacities, colTheme;

  function init() {
    columns = Math.floor(canvas.width / (fontSize * 8));
    drops = []; dropWords = []; dropSpeeds = []; dropOpacities = []; colTheme = [];
    for (let i = 0; i < columns; i++) {
      drops[i]         = Math.random() * -canvas.height;
      dropWords[i]     = pharmaWords[Math.floor(Math.random() * pharmaWords.length)];
      dropSpeeds[i]    = 0.3 + Math.random() * 0.7;
      dropOpacities[i] = 0.03 + Math.random() * 0.12;
      colTheme[i]      = colorThemes[Math.floor(Math.random() * colorThemes.length)];
    }
  }

  init();

  function draw() {
    ctx.fillStyle = 'rgba(0,0,0,0.04)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.font = fontSize + "px 'Courier New', monospace";

    for (let i = 0; i < columns; i++) {
      const word  = dropWords[i];
      const x     = i * (fontSize * 8);
      const y     = drops[i];
      const theme = colTheme[i];
      const op    = dropOpacities[i];

      for (let j = 0; j < word.length; j++) {
        const charY = y + j * (fontSize + 2);
        if (charY < 0 || charY > canvas.height) continue;
        const brightness = j === 0 ? 1 : Math.max(0.15, 1 - j * 0.1);
        if (j === 0) {
          ctx.fillStyle   = 'rgba(255,255,255,' + (op * 6) + ')';
          ctx.shadowColor = 'rgb(' + theme.r + ',' + theme.g + ',' + theme.b + ')';
          ctx.shadowBlur  = 8;
        } else {
          ctx.fillStyle  = 'rgba(' + theme.r + ',' + theme.g + ',' + theme.b + ',' + (op * brightness * 8) + ')';
          ctx.shadowBlur = 0;
        }
        ctx.fillText(word[j], x, charY);
      }

      ctx.shadowBlur = 0;
      drops[i] += dropSpeeds[i] * (fontSize + 2);

      if (drops[i] > canvas.height + word.length * (fontSize + 2)) {
        drops[i]         = -word.length * (fontSize + 2) * Math.random() * 3;
        dropWords[i]     = pharmaWords[Math.floor(Math.random() * pharmaWords.length)];
        dropSpeeds[i]    = 0.3 + Math.random() * 0.7;
        dropOpacities[i] = 0.03 + Math.random() * 0.12;
      }
    }
    requestAnimationFrame(draw);
  }

  draw();
})();
</script>
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
    
    books = ["Clinical Pharmacology Made Incredibly Easy (3rd Ed.).pdf"] # تأكد من وجود الملف
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

def ask_drugbrain(llm, v_store, query, is_table=False):
    docs = v_store.similarity_search(query, k=3)
    context = "\n".join([d.page_content for d in docs])
    table_instr = "\nهام: اعرض الأدوية في جدول Markdown (الدواء | الجرعة | ملاحظات)." if is_table else ""
    
    prompt = f"السياق: {context}\nالسؤال: {query}{table_instr}\nأجب باللهجة المصرية العامية كخبير صيدلي."
    return llm.invoke(prompt).content

# ====== 3. واجهة التطبيق (Frontend) ======
def main():
    # نظام التنظيف الذاتي
    if 'session_uses' not in st.session_state: st.session_state.session_uses = 0
    st.session_state.session_uses += 1
    if st.session_state.session_uses % 5 == 0: gc.collect()

    with st.sidebar:
        st.header("⚙️ إدارة النظام")
        if st.button("♻️ تنشيط الذاكرة (Reboot)"):
            st.cache_resource.clear()
            gc.collect()
            st.rerun()
        st.info("لو التطبيق تقل معاك، دوس هنا.")

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
                res = ask_drugbrain(llm, v_store, f"حلل الروشتة دي وطلع الأدوية: {raw_text}", is_table=True)
                st.markdown(f"<div class='report-card'><h3>🩺 التقرير:</h3>{res}</div>", unsafe_allow_html=True)

    with tab2:
        q = st.text_input("اسأل عن أي دواء أو حالة:")
        if q and st.button("🔍 بحث", key="b2"):
            res = ask_drugbrain(llm, v_store, q)
            st.markdown(f"<div class='report-card'>{res}</div>", unsafe_allow_html=True)

    with tab3:
        drugs = st.text_area("أدخل الأدوية لفحص التفاعلات:")
        if drugs and st.button("🚨 فحص", key="b3"):
            res = ask_drugbrain(llm, v_store, f"هل فيه تعارض بين: {drugs}؟", is_table=True)
            st.markdown(f"<div class='report-card'>{res}</div>", unsafe_allow_html=True)

    with tab4:
        f2 = st.file_uploader("ارفع صورة التحليل:", type=['jpg','png','jpeg'], key="lab")
        if f2 and st.button("🧬 تحليل النتائج", key="b4"):
            with st.spinner("🔍 جاري الفحص..."):
                reader = get_ocr_reader()
                raw_lab = " ".join(reader.readtext(np.array(Image.open(f2)), detail=0))
                gc.collect()
            with st.spinner("🩸 جاري كتابة التقرير..."):
                res = ask_drugbrain(llm, v_store, f"حلل التحليل ده: {raw_lab}")
                st.markdown(f"<div class='report-card'><h3>🩸 نتائج التحليل:</h3>{res}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
