import streamlit as st
import fitz # PyMuPDF
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize

# --- 0. Configuración Inicial y Descarga de NLTK ---
# Descarga directamente el recurso necesario 'punkt' de NLTK.
# Esta es la corrección que asegura la funcionalidad en Streamlit Cloud.
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    st.error(f"Error al inicializar NLTK (punkt): {e}")

# --- 1. Configuración y Carga de Modelo ---
# Modelo de lenguaje causal en español (GPT-2 Small)
MODEL_NAME = "datificate/gpt2-small-spanish"
# Umbral de Perplejidad: ajusta este valor. Un valor bajo (ej. 50) indica texto predecible (IA).
PERPLEXITY_THRESHOLD = 50 

@st.cache_resource
def load_model():
    """Carga el modelo y el tokenizador una sola vez."""
    try:
        st.info(f"Cargando modelo de IA: {MODEL_NAME}... Esto puede tardar unos segundos.")
        
        # Usamos try-except para manejar fallos de carga grandes
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        
        # Asegurarse de que el tokenizador tenga un token de pad
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return tokenizer, model
    except Exception as e:
        st.error(f"Error crítico al cargar el modelo de IA. Verifique el nombre o los recursos de memoria: {e}")
        return None, None

tokenizer, model = load_model()

# --- 2. Función de Detección de Perplejidad ---

def calculate_perplexity(text):
    """Calcula la perplejidad del texto usando el modelo CLM."""
    if not model or not tokenizer:
        return np.inf

    # Limpiar y preparar el texto para el modelo
    input_text = text.strip().replace('\n', ' ')
    if not input_text:
        return np.inf

    # Codificación del texto
    encodings = tokenizer(input_text, return_tensors='pt', truncation=True, padding=True)
    
    # Perplejidad = exponente de la pérdida (loss)
    with torch.no_grad():
        # Usamos labels=input_ids para calcular la pérdida del token siguiente
        loss = model(**encodings, labels=encodings.input_ids).loss
    
    # Calcula la perplejidad (e^loss)
    perplexity = torch.exp(loss).item()
    return perplexity

def analyze_text_for_ai(text):
    """Divide el texto en frases, las clasifica por perplejidad y calcula el porcentaje total."""
    
    # 1. División en frases
    # Usamos sent_tokenize de NLTK (ya descargado)
    sentences = sent_tokenize(text, language='spanish')
    
    results = []
    ai_sentence_count = 0

    # 2. Análisis por frase
    for sentence in sentences:
        if not sentence.strip():
            continue
        
        # Calcular la perplejidad de la frase
        ppl = calculate_perplexity(sentence)
        
        # Clasificación: Si la perplejidad es baja, es más probable que sea IA.
        is_ai = ppl < PERPLEXITY_THRESHOLD
        if is_ai:
            ai_sentence_count += 1
        
        results.append({
            "sentence": sentence,
            "perplexity": ppl,
            "is_ai": is_ai
        })
        
    total_sentences = len(results)
    if total_sentences == 0:
        return results, 0
        
    ai_percentage = (ai_sentence_count / total_sentences) * 100
    return results, ai_percentage

# --- 3. Interfaz de Streamlit ---

st.set_page_config(
    page_title="Detector de Texto IA en PDF (Español)",
    layout="centered"
)

st.title("🤖 Detector de Texto IA en PDF (Español)")
st.markdown("Sube un documento PDF para estimar la **probabilidad** de que haya sido generado por **Inteligencia Artificial** y **resaltar** las secciones sospechosas.")

uploaded_file = st.file_uploader("Sube tu archivo PDF aquí", type=["pdf"])

if uploaded_file is not None:
    # Mostrar el nombre del archivo
    st.markdown(f"**Archivo subido:** `{uploaded_file.name}`")
    
    # Extraer texto del PDF
    @st.cache_data
    def extract_text_from_pdf(file):
        """Extrae texto de un PDF usando PyMuPDF (fitz)."""
        try:
            # fitz necesita que el archivo se lea como bytes
            doc = fitz.open(stream=file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            st.error(f"Error al leer el PDF: {e}")
            return None

    # Botón de análisis
    if st.button("Analizar Documento", type="primary"):
        with st.spinner("Extrayendo texto y analizando contenido..."):
            extracted_text = extract_text_from_pdf(uploaded_file)

            if extracted_text:
                if len(extracted_text.strip()) < 50:
                    st.warning("El texto extraído es demasiado corto para un análisis confiable (mínimo 50 caracteres).")
                else:
                    analysis_results, ai_percentage = analyze_text_for_ai(extracted_text)
                    
                    # --- Resultados ---
                    st.header("Resultados del Análisis")
                    
                    col1, col2 = st.columns(2)
                    
                    # Columna 1: Porcentaje
                    col1.metric(
                        label="Probabilidad de Texto Generado por IA", 
                        value=f"{ai_percentage:.2f}%"
                    )
                    
                    # Columna 2: Umbral (Explicación)
                    col2.info(f"Umbral de Perplejidad: **<{PERPLEXITY_THRESHOLD}**. La baja perplejidad indica texto predecible.")
                    
                    st.divider()

                    # Texto Resaltado
                    st.subheader("Texto Analizado y Detecciones")
                    
                    # Construir el texto resaltado usando HTML/Markdown
                    highlighted_text = []
                    for item in analysis_results:
                        # Limpiar saltos de línea para que la frase se muestre continua
                        sentence = item['sentence'].replace('\n', ' ')
                        
                        if item['is_ai']:
                            # Usar HTML para resaltar con un fondo amarillo claro
                            highlighted_text.append(f"<mark style='background-color:#fff3cd;'>{sentence}</mark>")
                        else:
                            highlighted_text.append(sentence)
                    
                    # Unir y mostrar con HTML permitido (unsafe_allow_html=True)
                    st.markdown(
                        "".join(highlighted_text), 
                        unsafe_allow_html=True
                    )
            
            else:
                st.error("No se pudo extraer texto del documento PDF.")