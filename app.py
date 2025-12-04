import streamlit as st
import re
from pypdf import PdfReader
from docx import Document
import nltk

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Humanizador de Texto", layout="wide")

# --- CORRECCIÓN DEL ERROR DE NLTK ---
# Descargamos explícitamente ambos recursos necesarios para evitar el error "punkt_tab not found"
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# --- LISTA DE FRASES COMUNES DE IA (EN ESPAÑOL) ---
AI_PHRASES = [
    "En primer lugar", "En segundo término", "Por una parte... por otra", 
    "A continuación", "Asimismo", "De igual manera", "Del mismo modo",
    "En consecuencia", "Por lo tanto", "Así pues", "De ahí que",
    "En otras palabras", "Es decir", "Esto significa que",
    "A modo de ejemplo", "Para ilustrar esto", "Pongamos por caso",
    "Resulta fundamental", "Es de vital importancia", "Conviene subrayar",
    "Vale la pena recordar", "Cabe resaltar que", "Es relevante apuntar",
    "Como punto de partida", "En términos generales", "Desde una perspectiva amplia",
    "Históricamente", "Tradicionalmente", "En la actualidad",
    "En comparación con", "A diferencia de", "Por el contrario",
    "Si bien es cierto que", "A pesar de que", "Aun cuando",
    "Esto plantea la cuestión de", "Surge entonces la pregunta",
    "Desde un punto de vista crítico", "Analizando en profundidad",
    "En definitiva", "A modo de síntesis", "En esencia",
    "Para finalizar", "Como colofón", "En líneas generales",
    "El principal hallazgo es", "La conclusión principal radica en",
    "En última instancia", "A fin de cuentas",
    "Eficiente", "Óptimo", "Preciso", "Robusto", "Escalable",
    "Paradigma", "Marco conceptual", "Ecosistema",
    "Leveraje", "Sinergia", "Potenciar",
    "Interconectado", "Interdependiente", "Holístico",
    "Estado del arte", "Algoritmo", "Modelo predictivo", "Conjunto de datos",
    "Capacidad de generalización", "Procesamiento del lenguaje natural (PLN)",
    "Por lo general", "En la mayoría de los casos", "Suele ocurrir que",
    "Es probable que", "Podría considerarse", "Existe la posibilidad de",
    "Según los datos disponibles", "Basándonos en la información proporcionada",
    "Es recomendable", "Se sugiere", "Podría ser beneficioso",
    "Para responder a tu pregunta", "Me preguntas sobre", "Entiendo que buscas",
    "Voy a desglosarlo", "Permíteme explicarlo paso a paso",
    "¿Te gustaría que profundice en algún punto en particular?",
    "Es un tema complejo, pero intentaré simplificarlo"
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
            f'<span style="background-color: #ffd700; color: black; font-weight: bold; padding: 2px; border-radius: 3px;" title="Frase común de IA">{phrase}</span>', 
            highlighted_text
        )
    
    # Contamos cuántas frases de IA se encontraron para las métricas
    count_ai_phrases = sum(1 for phrase in AI_PHRASES if phrase in text.lower())
    
    # Usamos sent_tokenize solo para verificar que NLTK funciona, 
    # aunque en este MVP no estamos modificando la estructura de oraciones visualmente.
    try:
        sentences = nltk.tokenize.sent_tokenize(text)
        num_sentences = len(sentences)
    except Exception as e:
        num_sentences = 0
        print(f"Error en tokenización: {e}")
    
    return highlighted_text, count_ai_phrases, num_sentences

# --- INTERFAZ DE USUARIO (STREAMLIT) ---

st.title("🕵️ Detector y Humanizador de Textos")
st.markdown("""
### ¿Cómo funciona?
Sube tu documento y la herramienta resaltará patrones repetitivos.
* <span style="background-color: #ffd700; color: black; padding: 2px; border-radius: 3px;">**Amarillo**</span>: Conectores y "muletillas" que delatan a ChatGPT/Claude.
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
        
        st.success(f"Archivo **{uploaded_file.name}** cargado.")
        
        # Botón de análisis
        if st.button("🔍 Analizar Texto"):
            if not raw_text.strip():
                st.warning("El documento parece estar vacío o no se pudo leer el texto.")
            else:
                with st.spinner("Escaneando patrones..."):
                    html_result, count, num_sentences = analyze_and_highlight(raw_text)
                
                # Métricas
                col1, col2, col3 = st.columns(3)
                col1.metric("Palabras Totales", len(raw_text.split()))
                col2.metric("Oraciones", num_sentences)
                col3.metric("Frases 'Robóticas'", count, delta_color="inverse")
                
                st.markdown("---")
                st.subheader("📝 Resultado del Análisis")
                st.info("Sugerencia: Reescribe las partes amarillas usando un lenguaje más coloquial o directo.")
                
                # Caja con el texto resaltado (Scrollable)
                st.markdown(
                    f"""
                    <div style="
                        padding: 20px; 
                        border: 1px solid #ccc; 
                        border-radius: 10px; 
                        background-color: white; 
                        color: #333; 
                        line-height: 1.8; 
                        height: 500px; 
                        overflow-y: scroll;
                        font-family: Arial, sans-serif;
                    ">
                        {html_result}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")

else:
    st.info("👈 Sube un archivo TXT, PDF o DOCX desde el menú lateral.")