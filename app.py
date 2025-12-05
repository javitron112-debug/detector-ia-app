import streamlit as st
import fitz  # PyMuPDF
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 0. Configuración Inicial ---
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    st.error(f"Error al inicializar NLTK: {e}")

# --- 1. Configuración de Modelo ---
MODEL_NAME = "datificate/gpt2-small-spanish"
PERPLEXITY_THRESHOLD = 50

@st.cache_resource
def load_model():
    """Carga el modelo y tokenizador."""
    try:
        with st.spinner(f"Cargando modelo {MODEL_NAME}..."):
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()  # Modo evaluación
                
            return tokenizer, model, device
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None, None, None

tokenizer, model, device = load_model()

# --- 2. Funciones de Análisis Mejoradas ---

def calculate_perplexity(text, max_length=512):
    """Calcula perplejidad con manejo mejorado de texto largo."""
    if not model or not tokenizer or not text.strip():
        return np.inf

    input_text = text.strip().replace('\n', ' ')
    
    try:
        encodings = tokenizer(
            input_text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=max_length,
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**encodings, labels=encodings.input_ids)
            loss = outputs.loss
        
        perplexity = torch.exp(loss).item()
        return perplexity
    except Exception as e:
        st.warning(f"Error calculando perplejidad: {e}")
        return np.inf

def analyze_text_for_ai(text, threshold=PERPLEXITY_THRESHOLD):
    """Analiza texto con estadísticas detalladas."""
    sentences = sent_tokenize(text)
    
    results = []
    ai_sentence_count = 0
    perplexities = []

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, sentence in enumerate(sentences):
        if not sentence.strip():
            continue
        
        # Actualizar progreso
        progress = (idx + 1) / len(sentences)
        progress_bar.progress(progress)
        status_text.text(f"Analizando frase {idx + 1} de {len(sentences)}...")
        
        ppl = calculate_perplexity(sentence)
        perplexities.append(ppl)
        
        is_ai = ppl < threshold
        if is_ai:
            ai_sentence_count += 1
        
        results.append({
            "sentence": sentence,
            "perplexity": ppl,
            "is_ai": is_ai,
            "word_count": len(sentence.split())
        })
    
    progress_bar.empty()
    status_text.empty()
    
    total_sentences = len(results)
    if total_sentences == 0:
        return results, 0, {}
    
    ai_percentage = (ai_sentence_count / total_sentences) * 100
    
    # Estadísticas adicionales
    valid_perplexities = [p for p in perplexities if p != np.inf]
    
    stats = {
        "total_sentences": total_sentences,
        "ai_sentences": ai_sentence_count,
        "human_sentences": total_sentences - ai_sentence_count,
        "avg_perplexity": np.mean(valid_perplexities) if valid_perplexities else 0,
        "median_perplexity": np.median(valid_perplexities) if valid_perplexities else 0,
        "min_perplexity": np.min(valid_perplexities) if valid_perplexities else 0,
        "max_perplexity": np.max(valid_perplexities) if valid_perplexities else 0,
    }
    
    return results, ai_percentage, stats

def create_perplexity_chart(results, threshold):
    """Crea gráfico de distribución de perplejidad usando matplotlib."""
    perplexities = [r['perplexity'] for r in results if r['perplexity'] != np.inf]
    
    if not perplexities:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Histograma
    ax.hist(perplexities, bins=30, color='#636EFA', alpha=0.7, edgecolor='black')
    
    # Línea de umbral
    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Umbral: {threshold}')
    
    ax.set_xlabel('Perplejidad', fontsize=12)
    ax.set_ylabel('Frecuencia', fontsize=12)
    ax.set_title('Distribución de Perplejidad por Frase', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

# --- 3. Interfaz Mejorada ---

st.set_page_config(
    page_title="Detector de Texto IA en PDF",
    page_icon="🤖",
    layout="wide"
)

# Sidebar con configuración
with st.sidebar:
    st.header("⚙️ Configuración")
    
    custom_threshold = st.slider(
        "Umbral de Perplejidad",
        min_value=10,
        max_value=200,
        value=PERPLEXITY_THRESHOLD,
        step=5,
        help="Valores más bajos = más estricto en la detección de IA"
    )
    
    st.divider()
    
    st.markdown("""
    ### ℹ️ Cómo funciona
    
    Este detector usa un modelo de lenguaje para calcular la **perplejidad** de cada frase:
    
    - **Baja perplejidad** (<50): Texto predecible, posiblemente generado por IA
    - **Alta perplejidad** (>50): Texto más natural y variado
    
    ⚠️ **Nota**: Esta es una estimación probabilística, no una certeza absoluta.
    """)
    
    if device:
        st.info(f"🖥️ Dispositivo: {device.type.upper()}")

# Header principal
st.title("🤖 Detector de Texto IA en PDF")
st.markdown("""
Analiza documentos PDF para identificar secciones que podrían haber sido generadas por Inteligencia Artificial.
""")

uploaded_file = st.file_uploader(
    "📄 Sube tu archivo PDF",
    type=["pdf"],
    help="El archivo será analizado localmente"
)

if uploaded_file is not None:
    st.success(f"✅ Archivo cargado: **{uploaded_file.name}**")
    
    @st.cache_data
    def extract_text_from_pdf(file):
        """Extrae texto del PDF."""
        try:
            doc = fitz.open(stream=file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            st.error(f"Error al leer el PDF: {e}")
            return None

    col1, col2 = st.columns([1, 4])
    
    with col1:
        analyze_button = st.button("🔍 Analizar", type="primary", use_container_width=True)
    
    if analyze_button:
        if model is None or tokenizer is None:
            st.error("❌ El modelo no se cargó correctamente.")
        else:
            extracted_text = extract_text_from_pdf(uploaded_file)

            if extracted_text:
                if len(extracted_text.strip()) < 50:
                    st.warning("⚠️ El texto extraído es demasiado corto (mínimo 50 caracteres).")
                else:
                    analysis_results, ai_percentage, stats = analyze_text_for_ai(
                        extracted_text, 
                        threshold=custom_threshold
                    )
                    
                    # --- Resultados ---
                    st.header("📊 Resultados del Análisis")
                    
                    # Métricas principales
                    metric_cols = st.columns(4)
                    
                    metric_cols[0].metric(
                        "Probabilidad IA",
                        f"{ai_percentage:.1f}%"
                    )
                    
                    metric_cols[1].metric(
                        "Total Frases",
                        stats['total_sentences']
                    )
                    
                    metric_cols[2].metric(
                        "Frases IA",
                        stats['ai_sentences']
                    )
                    
                    metric_cols[3].metric(
                        "Perplejidad Media",
                        f"{stats['avg_perplexity']:.1f}"
                    )
                    
                    st.divider()
                    
                    # Tabs para diferentes vistas
                    tab1, tab2, tab3 = st.tabs(["📝 Texto Resaltado", "📈 Estadísticas", "📋 Detalle por Frase"])
                    
                    with tab1:
                        st.subheader("Texto Analizado")
                        st.caption("🟨 Amarillo = Posible IA | ⬜ Blanco = Posible Humano")
                        
                        highlighted_text = []
                        for item in analysis_results:
                            sentence = item['sentence'].replace('\n', ' ')
                            ppl = item['perplexity']
                            
                            if item['is_ai']:
                                tooltip = f"Perplejidad: {ppl:.2f}"
                                highlighted_text.append(
                                    f"<mark style='background-color:#fff3cd;' title='{tooltip}'>{sentence}</mark> "
                                )
                            else:
                                highlighted_text.append(f"{sentence} ")
                        
                        st.markdown(
                            "".join(highlighted_text),
                            unsafe_allow_html=True
                        )
                    
                    with tab2:
                        st.subheader("Distribución de Perplejidad")
                        
                        chart = create_perplexity_chart(analysis_results, custom_threshold)
                        if chart:
                            st.pyplot(chart)
                        
                        st.divider()
                        
                        # Estadísticas adicionales
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 📊 Estadísticas Generales")
                            st.write(f"- **Perplejidad mínima:** {stats['min_perplexity']:.2f}")
                            st.write(f"- **Perplejidad máxima:** {stats['max_perplexity']:.2f}")
                            st.write(f"- **Perplejidad mediana:** {stats['median_perplexity']:.2f}")
                        
                        with col2:
                            st.markdown("### 🎯 Clasificación")
                            st.write(f"- **Frases posible IA:** {stats['ai_sentences']}")
                            st.write(f"- **Frases posible humano:** {stats['human_sentences']}")
                            st.write(f"- **Porcentaje IA:** {ai_percentage:.2f}%")
                        
                        # Tabla de resumen
                        st.subheader("📋 Resumen en Tabla")
                        df = pd.DataFrame({
                            'Métrica': ['Total frases', 'Frases IA', 'Frases Humano', 'Perplejidad Media', 'Perplejidad Mediana'],
                            'Valor': [
                                stats['total_sentences'],
                                stats['ai_sentences'],
                                stats['human_sentences'],
                                f"{stats['avg_perplexity']:.2f}",
                                f"{stats['median_perplexity']:.2f}"
                            ]
                        })
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    with tab3:
                        st.subheader("Detalle por Frase")
                        
                        # Filtros
                        filter_col1, filter_col2 = st.columns(2)
                        with filter_col1:
                            show_filter = st.selectbox(
                                "Mostrar:",
                                ["Todas", "Solo IA", "Solo Humano"]
                            )
                        
                        filtered_results = analysis_results
                        if show_filter == "Solo IA":
                            filtered_results = [r for r in analysis_results if r['is_ai']]
                        elif show_filter == "Solo Humano":
                            filtered_results = [r for r in analysis_results if not r['is_ai']]
                        
                        for idx, item in enumerate(filtered_results, 1):
                            tipo = '🤖 IA' if item['is_ai'] else '👤 Humano'
                            with st.expander(
                                f"Frase {idx} - {tipo} (Perplejidad: {item['perplexity']:.2f})",
                                expanded=False
                            ):
                                st.write(item['sentence'])
                                st.caption(f"📝 Palabras: {item['word_count']}")
            else:
                st.error("❌ No se pudo extraer texto del PDF.")
else:
    st.info("👆 Sube un archivo PDF para comenzar el análisis.")