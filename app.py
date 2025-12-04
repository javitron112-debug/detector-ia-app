import streamlit as st
import re
from pypdf import PdfReader
from docx import Document
import nltk

# Descargar tokenizador de oraciones si es la primera vez
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Humanizador de Texto", layout="wide")

# --- LISTA DE FRASES COMUNES DE IA (EN ESPAÑOL) ---
# Los detectores buscan estas transiciones excesivamente formales y estructuradas.
AI_PHRASES = [
    "es importante destacar", "en conclusión", "por otro lado", 
    "cabe mencionar", "en resumen", "además", "sin embargo", 
    "es crucial", "en el contexto de", "un papel fundamental",
    "transformación digital", "amplia gama de", "meticulosamente"
]

# --- FUNCIONES DE EXTRACCIÓN ---

def read_txt(file):
    return str(file.read(), "utf-8")

def read_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

def read_docx(file):
    doc = Document(file)
    text = []
    for para in doc.paragraphs:
        text.append(para.text)
    return "\n".join(text)

# --- MOTOR DE ANÁLISIS ---

def analyze_and_highlight(text):
    """
    Analiza el texto buscando patrones de IA y devuelve HTML resaltado.
    """
    highlighted_text = text
    
    # 1. Resaltar frases comunes de IA (Color Amarillo)
    # Usamos Regex para reemplazar sin importar mayúsculas/minúsculas
    for phrase in AI_PHRASES:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        # El <span> añade el fondo amarillo
        highlighted_text = pattern.sub(
            f'<span style="background-color: #ffd700; color: black; font-weight: bold;" title="Frase común de IA">{phrase}</span>', 
            highlighted_text
        )
    
    # 2. Análisis de Monotonía (Rafagosidad baja)
    # Si una oración es muy larga y compleja, a veces es señal de IA.
    sentences = nltk.tokenize.sent_tokenize(text)
    
    # Reconstruimos el texto procesando oraciones
    # Nota: Este es un método simplificado de visualización. 
    # Para producción, se debe reconstruir con cuidado para no romper el HTML anterior.
    
    count_ai_phrases = sum(1 for phrase in AI_PHRASES if phrase in text.lower())
    
    return highlighted_text, count_ai_phrases

# --- INTERFAZ DE USUARIO (STREAMLIT) ---

st.title("🕵️ Detector y Humanizador de Textos")
st.markdown("""
Sube tu documento (.txt, .pdf, .docx). La aplicación resaltará:
* <span style="background-color: #ffd700; color: black;">Amarillo</span>: Frases "muletilla" típicas de la IA.
""", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Sube tu archivo", type=["txt", "pdf", "docx"])

if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1]
    raw_text = ""

    # Procesar archivo según tipo
    try:
        if file_type == "txt":
            raw_text = read_txt(uploaded_file)
        elif file_type == "pdf":
            raw_text = read_pdf(uploaded_file)
        elif file_type == "docx":
            raw_text = read_docx(uploaded_file)
        
        st.success(f"Archivo '{uploaded_file.name}' procesado correctamente.")
        
        # Botón de análisis
        if st.button("Analizar Texto"):
            with st.spinner("Buscando patrones de IA..."):
                html_result, count = analyze_and_highlight(raw_text)
            
            # Métricas rápidas
            col1, col2 = st.columns(2)
            col1.metric("Palabras Totales", len(raw_text.split()))
            col2.metric("Frases de IA detectadas", count)
            
            st.markdown("### Resultado del Análisis")
            st.info("Edita las partes resaltadas para aumentar la 'Rafagosidad' y naturalidad del texto.")
            
            # Caja con el texto resaltado
            st.markdown(
                f'<div style="padding: 20px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9; color: #333; line-height: 1.6;">{html_result}</div>', 
                unsafe_allow_html=True
            )

    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")

else:
    st.info("Por favor, sube un archivo desde el menú lateral para comenzar.")