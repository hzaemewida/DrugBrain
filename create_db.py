from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

print("⏳ جاري قراءة الكتب الطبية...")
# تأكد إن أسماء الكتب دي موجودة معاك في نفس الفولدر
books_list = [
    "Clinical Pharmacology Made Incredibly Easy (3rd Ed.).pdf",
    "Book_2.pdf",
    "Book_3.pdf"
]

all_docs = []
for path in books_list:
    if os.path.exists(path):
        print(f"✅ جاري تحميل {path}...")
        loader = PyPDFLoader(path)
        all_docs.extend(loader.load())
    else:
        print(f"❌ الملف {path} غير موجود!")

if all_docs:
    print("✂️ جاري تقطيع النصوص...")
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(all_docs)

    print("🧠 جاري تحويل النصوص لمعادلات وبناء قاعدة البيانات (FAISS)...")
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(split_docs, embed_model)

    # هنا هيعمل فولدر جديد اسمه faiss_index ويحفظ جواه الشغل
    vector_store.save_local("faiss_index")
    print("🎉 تم بنجاح! تم إنشاء مجلد 'faiss_index'. ارفع هذا المجلد على جيت هاب.")
else:
    print("⚠️ لم يتم العثور على أي ملفات PDF لمعالجتها.")
