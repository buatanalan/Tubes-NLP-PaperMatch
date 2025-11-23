import streamlit as st
import os
import json
import torch
import shutil
import warnings
from huggingface_hub import snapshot_download, hf_hub_download
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
import torch.nn.functional as F

# ==========================================
# KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Paper Topic & Related Work Gen",
    page_icon="📚",
    layout="wide"
)

CONFIG = {
    "CHROMA_REPO": "iskandarmrp/nlp-papermatch-dataset",
    "CHROMA_DIR": "./chroma_download",
    "CLASSIFIER_MODEL": "iskandarmrp/distilbert-lora-paper-topic-classification",
    "GENERATOR_MODEL": "Alan43/related_works_generation_model",
    "GENERATOR_DIR": "./model_local_final",
    "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "LABEL_MAPPING_FILE": "label_mapping.json"
}

# ==========================================
# FUNGSI UTILITAS (CACHED)
# ==========================================

@st.cache_resource
def setup_chroma_db():
    chroma_path = os.path.join(CONFIG["CHROMA_DIR"], "chroma_db")
    if os.path.exists(chroma_path) and len(os.listdir(chroma_path)) > 0:
        return chroma_path
    else:
        with st.spinner("Downloading ChromaDB..."):
            snapshot_download(
                repo_id=CONFIG["CHROMA_REPO"],
                repo_type="dataset",
                local_dir=CONFIG["CHROMA_DIR"],
                allow_patterns="chroma_db/*",
                local_dir_use_symlinks=False
            )
    return chroma_path

@st.cache_data
def get_label_mapping():
    if os.path.exists(CONFIG["LABEL_MAPPING_FILE"]):
        with open(CONFIG["LABEL_MAPPING_FILE"], "r") as f:
            return json.load(f)
    
    try:
        file_path = hf_hub_download(repo_id=CONFIG["CLASSIFIER_MODEL"], filename="label_mapping.json")
        with open(file_path, "r") as f:
            mapping = json.load(f)
        with open(CONFIG["LABEL_MAPPING_FILE"], "w") as f:
            json.dump(mapping, f)
        return mapping
    except Exception as e:
        return {}

# ==========================================
# KELAS PIPELINE (DIMODIFIKASI UNTUK STREAMLIT)
# ==========================================

class ResearchPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.label_mapping = get_label_mapping()
        self.id2label = {v: k for k, v in self.label_mapping.items()}
        
        # Load Classifier
        self.cls_config = PeftConfig.from_pretrained(CONFIG["CLASSIFIER_MODEL"])
        self.cls_tokenizer = AutoTokenizer.from_pretrained(self.cls_config.base_model_name_or_path)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.cls_config.base_model_name_or_path,
            num_labels=len(self.label_mapping) if self.label_mapping else 20,
            ignore_mismatched_sizes=True
        )
        self.classifier = PeftModel.from_pretrained(base_model, CONFIG["CLASSIFIER_MODEL"])
        self.classifier.to("cpu") # Hemat VRAM untuk generator
        self.classifier.eval()

        # Load Chroma
        chroma_path = setup_chroma_db()
        embedding_fn = HuggingFaceEmbeddings(
            model_name=CONFIG["EMBEDDING_MODEL"],
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vector_db = Chroma(
            persist_directory=chroma_path,
            embedding_function=embedding_fn,
            collection_name="paper_abstracts"
        )

        # Load Generator (Hanya jika ada GPU)
        self.generator = None
        self.gen_tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # Bersihkan cache GPU sebelum load model besar
            try:
                if not os.path.exists(CONFIG["GENERATOR_DIR"]):
                      snapshot_download(repo_id=CONFIG["GENERATOR_MODEL"], local_dir=CONFIG["GENERATOR_DIR"])

                # Konfigurasi persis seperti script manual yang berhasil
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    llm_int8_enable_fp32_cpu_offload=False 
                )
                self.gen_tokenizer = AutoTokenizer.from_pretrained(CONFIG["GENERATOR_DIR"])
                if self.gen_tokenizer.pad_token_id is None:
                    self.gen_tokenizer.pad_token_id = self.gen_tokenizer.eos_token_id
                
                # Suppress warning tentang quantization config yang duplikat
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, message=".*quantization_config.*")
                    
                    self.generator = AutoModelForCausalLM.from_pretrained(
                        CONFIG["GENERATOR_DIR"],
                        quantization_config=bnb_config,
                        device_map="cuda:0", # Paksa ke GPU 0 seperti script manual
                        trust_remote_code=True,
                        local_files_only=True
                    )
            except Exception as e:
                st.error(f"Gagal memuat Generator: {e}")

    def predict_topic(self, text):
        inputs = self.cls_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256, padding="max_length"
        ).to("cpu") # Classifier run on CPU
        
        with torch.no_grad():
            outputs = self.classifier(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            pred_id = torch.argmax(probs, dim=1).item()
        
        label = self.id2label.get(pred_id, str(pred_id))
        return label, pred_id

    def retrieve_documents(self, query, topic_id, k=3):
        results = self.vector_db.similarity_search(
            query, k=k, filter={"label": topic_id}
        )
        if not results:
            results = self.vector_db.similarity_search(query, k=k)
        return results

    def generate_related_work(self, query_abstract, retrieved_docs):
        if not self.generator:
            return "Generator model tidak tersedia (Mode CPU atau Gagal Load).", ""

        references_text = ""
        for i, doc in enumerate(retrieved_docs):
            title = doc.metadata.get("title", "Unknown Title")
            abstract = doc.page_content
            references_text += f"@cite_{i+1}: Title: {title}\nAbstract: {abstract}\n"

        full_input = f"Current Abstract:\n{query_abstract}\n\nReferences:{references_text}"
        system_msg = "You are an academic writing assistant. Write a 'Related Work' section based on the provided text. The input contains the Current Abstract followed by References (marked with @cite_n). Synthesize these references and highlight the novelty of the Current Abstract."
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>

{full_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        inputs = self.gen_tokenizer(prompt, return_tensors="pt").to("cuda")

        generation_kwargs = {
            "max_new_tokens": 300,
            "min_length": 60,
            "num_beams": 1,
            "do_sample": True,
            "temperature": 0.1,
            "top_p": 0.9,
            "no_repeat_ngram_size": 0,
            "repetition_penalty": 1.1,
        }

        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,                         # Unpack input_ids & attention_mask
                **generation_kwargs,              # Unpack config di atas
                use_cache=True,
                eos_token_id=self.gen_tokenizer.eos_token_id,
                pad_token_id=self.gen_tokenizer.pad_token_id, # Gunakan pad token yang benar
            )
        
        generated_text = self.gen_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        return generated_text, references_text

# ==========================================
# INISIALISASI MODEL (CACHED)
# ==========================================
@st.cache_resource(show_spinner="Memuat model AI yang berat, mohon tunggu sebentar...")
def load_pipeline():
    return ResearchPipeline()

# ==========================================
# UI UTAMA
# ==========================================

st.title("📚 AI Research Assistant")
st.markdown("Masukkan abstrak paper Anda untuk **klasifikasi topik**, **pencarian referensi**, dan **pembuatan draft Related Works**.")

# Load Pipeline
try:
    pipeline = load_pipeline()
    st.sidebar.success("✅ Model Loaded Successfully")
    st.sidebar.info(f"Device: {pipeline.device}")
except Exception as e:
    st.error(f"Gagal memuat pipeline: {e}")
    st.stop()

# Input User
user_abstract = st.text_area("Masukkan Abstract di sini:", height=200, placeholder="Paste abstract text here...")

if st.button("🚀 Analisis & Generate", type="primary"):
    if not user_abstract:
        st.warning("Mohon isi abstrak terlebih dahulu.")
    else:
        # Container untuk hasil
        col1, col2 = st.columns([1, 1.5])
        
        with st.status("Sedang memproses...", expanded=True) as status:
            
            # 1. Klasifikasi
            status.write("🔍 Mengidentifikasi Topik...")
            topic, topic_id = pipeline.predict_topic(user_abstract)
            
            # 2. Retrieval
            status.write("📚 Mencari Referensi Relevan (RAG)...")
            docs = pipeline.retrieve_documents(user_abstract, topic_id, k=3)
            
            # 3. Generation
            status.write("📝 Menulis Related Work Section...")
            result_text, _ = pipeline.generate_related_work(user_abstract, docs)
            
            status.update(label="Selesai!", state="complete", expanded=False)

        # === TAMPILKAN HASIL ===
        
        # Kolom Kiri: Metadata & Referensi
        with col1:
            st.subheader("📊 Hasil Analisis")
            st.info(f"**Prediksi Topik:** {topic}")
            st.caption(f"Topic ID: {topic_id}")
            
            st.divider()
            st.subheader("📖 Referensi Terkait")
            for i, doc in enumerate(docs):
                title = doc.metadata.get("title", "No Title")
                # Jika ada ID di metadata, tampilkan. Jika tidak, gunakan index.
                paper_id = doc.metadata.get("id", f"Ref-{i+1}") 
                
                with st.expander(f"📄 {title}"):
                    st.caption(f"**ID:** {paper_id}")
                    st.markdown(f"_{doc.page_content[:300]}..._")
        
        # Kolom Kanan: Hasil Generasi
        with col2:
            st.subheader("✍️ Generated Related Work")
            if "Generator model tidak tersedia" in result_text:
                st.warning(result_text)
            else:
                st.markdown(result_text)
                st.download_button(
                    label="Download Text",
                    data=result_text,
                    file_name="related_works.md",
                    mime="text/markdown"
                )

# # Footer
# st.markdown("---")
# st.caption("Ditenagai oleh DistilBERT LoRA, ChromaDB, dan Llama-3.")
# ```

# ### Cara Menjalankan

# 1.  Simpan file di atas sebagai `app.py`.
# 2.  Buka terminal/command prompt.
# 3.  Jalankan perintah:
#     ```bash
#     streamlit run app.py