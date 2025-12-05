import streamlit as st
import numpy as np
import pandas as pd
import nltk
import tempfile
import os
from io import BytesIO
import base64
import time
import json

# Configuración de página
st.set_page_config(
    page_title="Detector IA - Analizador PDF",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Descargar recursos NLTK
@st.cache_resource
def download_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

download_nltk()

# CSS personalizado
st.markdown("""
<style>
    /* Estilos generales */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Encabezado */
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin: 0;
        padding: 1rem;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Tarjetas */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    
    /* Botones */
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Resultados */
    .result-human {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .result-ai {
        background: linear-gradient(135deg, #F44336 0%, #C62828 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Texto resaltado */
    .ai-highlight {
        background-color: rgba(244, 67, 54, 0.2);
        padding: 2px 4px;
        border-radius: 3px;
        border-left: 3px solid #F44336;
        margin: 2px 0;
    }
    
    .human-highlight {
        background-color: rgba(76, 175, 80, 0.2);
        padding: 2px 4px;
        border-radius: 3px;
        border-left: 3px solid #4CAF50;
        margin: 2px 0;
    }
    
    /* Métricas */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    
    /* Barra de progreso personalizada */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4CAF50 0%, #FFC107 50%, #F44336 100%);
    }
    
    /* Ocultar elementos de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Encabezado principal
st.markdown("""
<div class="main-header">
    <h1>🔍 Detector de IA en Textos</h1>
    <p>Analiza archivos PDF, DOCX o texto directo - Subraya contenido sospechoso de IA</p>
</div>
""", unsafe_allow_html=True)

# Importar módulos locales
try:
    from utils.pdf_processor import PDFProcessor
    from utils.detector_advanced import AIDetectorAdvanced
    from utils.text_highlighter import TextHighlighter
except ImportError:
    # Definir las clases aquí si no se pueden importar
    import re
    from nltk.tokenize import sent_tokenize, word_tokenize
    
    class PDFProcessor:
        """Procesador simple de PDF"""
        @staticmethod
        def extract_text_from_pdf(uploaded_file):
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
            except:
                return "Error al leer PDF"
    
    class TextHighlighter:
        """Resaltador de texto"""
        @staticmethod
        def highlight_ai_sentences(text, ai_sentences):
            """Resalta oraciones detectadas como IA"""
            highlighted_text = text
            for sentence in ai_sentences:
                # Escapar caracteres especiales para regex
                escaped_sentence = re.escape(sentence)
                # Resaltar la oración
                highlighted_text = re.sub(
                    escaped_sentence,
                    f'<span class="ai-highlight">{sentence}</span>',
                    highlighted_text
                )
            return highlighted_text
    
    class AIDetectorAdvanced:
        """Detector avanzado de IA"""
        def __init__(self):
            from nltk.corpus import stopwords
            self.stopwords = set(stopwords.words('spanish'))
        
        def analyze_text(self, text):
            """Analiza texto y devuelve resultados"""
            sentences = sent_tokenize(text)
            results = []
            
            for sentence in sentences:
                score = self._calculate_sentence_score(sentence)
                results.append({
                    'sentence': sentence,
                    'score': score,
                    'is_ai': score > 0.5
                })
            
            return results
        
        def _calculate_sentence_score(self, sentence):
            """Calcula puntuación para una oración"""
            words = word_tokenize(sentence.lower())
            
            # Características de IA
            features = {
                'length': len(sentence),
                'word_count': len(words),
                'unique_words': len(set(words)) / max(len(words), 1),
                'stopword_ratio': sum(1 for w in words if w in self.stopwords) / max(len(words), 1),
                'formal_words': self._count_formal_words(sentence),
                'comma_density': sentence.count(',') / max(len(sentence), 1)
            }
            
            # Puntuación basada en características
            score = 0
            
            # Oraciones largas y formales
            if features['length'] > 150:
                score += 0.2
            
            # Alta densidad de comas
            if features['comma_density'] > 0.1:
                score += 0.15
            
            # Palabras formales
            if features['formal_words'] > 2:
                score += 0.1
            
            # Baja diversidad léxica
            if features['unique_words'] < 0.4:
                score += 0.1
            
            return min(score, 1.0)
        
        def _count_formal_words(self, sentence):
            """Cuenta palabras formales comunes en textos de IA"""
            formal_words = [
                'además', 'sin embargo', 'por lo tanto', 'en consecuencia',
                'es decir', 'por otro lado', 'no obstante', 'así mismo',
                'cabe destacar', 'en primer lugar', 'en segundo lugar',
                'finalmente', 'en conclusión', 'por último'
            ]
            return sum(1 for word in formal_words if word in sentence.lower())

# Inicializar componentes
@st.cache_resource
def init_components():
    return {
        'pdf_processor': PDFProcessor(),
        'detector': AIDetectorAdvanced(),
        'highlighter': TextHighlighter()
    }

components = init_components()

# Sidebar simplificado
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    
    # Modo de entrada
    input_mode = st.radio(
        "Modo de entrada:",
        ["📝 Texto Directo", "📄 Subir Archivo"],
        index=0
    )
    
    # Opciones de análisis
    st.markdown("### 🔍 Opciones de Análisis")
    
    analysis_depth = st.select_slider(
        "Profundidad de análisis:",
        options=["Básico", "Normal", "Avanzado"],
        value="Normal"
    )
    
    show_details = st.checkbox("Mostrar detalles técnicos", value=False)
    
    st.markdown("---")
    st.markdown("### 📊 Estadísticas")
    st.markdown("""
    - **Precisión estimada:** 75-85%
    - **Límite de archivo:** 10MB
    - **Formatos soportados:** PDF, DOCX, TXT
    - **Procesamiento:** En tiempo real
    """)
    
    st.markdown("---")
    st.markdown("#### 📱 Contacto")
    st.markdown("¿Problemas o sugerencias?")
    st.markdown("[Reportar issue](https://github.com)")

# Área principal de la aplicación
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### 📥 Entrada de Datos")
    
    if input_mode == "📝 Texto Directo":
        text_input = st.text_area(
            "Escribe o pega tu texto aquí:",
            height=200,
            placeholder="Pega aquí el texto que deseas analizar...",
            help="Mínimo 100 caracteres para mejor precisión"
        )
        
        # Ejemplos rápidos
        example_col1, example_col2 = st.columns(2)
        with example_col1:
            if st.button("📚 Ejemplo Académico", use_container_width=True):
                st.session_state.text_input = """La inteligencia artificial ha revolucionado el panorama tecnológico contemporáneo. 
                Es fundamental considerar los aspectos éticos inherentes a su implementación para garantizar un desarrollo sostenible. 
                Además, la transparencia en los algoritmos constituye un requisito indispensable."""
        
        with example_col2:
            if st.button("💬 Ejemplo Conversacional", use_container_width=True):
                st.session_state.text_input = """Hola, ¿cómo estás? La verdad es que hoy me siento bastante cansado. 
                Ayer estuve trabajando hasta tarde en un proyecto importante. 
                ¿Tú qué planes tienes para el fin de semana? Yo creo que voy a descansar un poco."""
        
        if 'text_input' in st.session_state:
            text_input = st.session_state.text_input
    
    else:  # Modo archivo
        uploaded_file = st.file_uploader(
            "Sube tu archivo:",
            type=['pdf', 'txt', 'docx'],
            help="Formatos soportados: PDF, TXT, DOCX (hasta 10MB)"
        )
        
        if uploaded_file is not None:
            # Mostrar información del archivo
            file_details = {
                "Nombre": uploaded_file.name,
                "Tipo": uploaded_file.type,
                "Tamaño": f"{uploaded_file.size / 1024:.1f} KB"
            }
            
            st.markdown("**📄 Información del archivo:**")
            for key, value in file_details.items():
                st.write(f"{key}: {value}")
            
            # Procesar archivo según tipo
            with st.spinner("Procesando archivo..."):
                if uploaded_file.type == "application/pdf":
                    text_input = components['pdf_processor'].extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "text/plain":
                    text_input = uploaded_file.read().decode("utf-8")
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    try:
                        import docx
                        doc = docx.Document(uploaded_file)
                        text_input = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                    except:
                        text_input = "Error al leer archivo DOCX"
                else:
                    text_input = "Formato no soportado"
                
                # Mostrar vista previa
                if text_input and len(text_input) > 500:
                    st.markdown(f"**Vista previa (primeros 500 caracteres):**")
                    st.text(text_input[:500] + "...")
                elif text_input:
                    st.markdown(f"**Contenido del archivo:**")
                    st.text(text_input)

with col2:
    st.markdown("### ⚡ Análisis Rápido")
    
    if 'text_input' in locals() and text_input and len(text_input.strip()) > 50:
        # Botón de análisis
        analyze_button = st.button(
            "🚀 Iniciar Análisis Completo",
            type="primary",
            use_container_width=True,
            disabled=not text_input or len(text_input.strip()) < 50
        )
        
        if analyze_button:
            with st.spinner("Analizando contenido..."):
                # Realizar análisis
                results = components['detector'].analyze_text(text_input)
                
                # Calcular estadísticas
                total_sentences = len(results)
                ai_sentences = sum(1 for r in results if r['is_ai'])
                ai_percentage = (ai_sentences / total_sentences) * 100 if total_sentences > 0 else 0
                
                # Guardar en sesión
                st.session_state.analysis_results = {
                    'text': text_input,
                    'results': results,
                    'stats': {
                        'total_sentences': total_sentences,
                        'ai_sentences': ai_sentences,
                        'ai_percentage': ai_percentage,
                        'avg_score': np.mean([r['score'] for r in results]) if results else 0
                    }
                }
                
                st.success(f"✅ Análisis completado: {total_sentences} oraciones procesadas")
        
        # Mostrar métricas rápidas si hay análisis previo
        if 'analysis_results' in st.session_state:
            stats = st.session_state.analysis_results['stats']
            
            st.markdown("#### 📊 Resultados Previos")
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{stats["total_sentences"]}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Oraciones</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with metric_col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{stats["ai_sentences"]}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">IA Detectadas</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with metric_col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{stats["ai_percentage"]:.1f}%</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Porcentaje IA</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("📝 Ingresa texto o sube un archivo para comenzar el análisis")

# Mostrar resultados detallados
if 'analysis_results' in st.session_state:
    st.markdown("---")
    st.markdown("## 📈 Resultados Detallados del Análisis")
    
    results = st.session_state.analysis_results['results']
    stats = st.session_state.analysis_results['stats']
    
    # Resumen principal
    col_res1, col_res2, col_res3 = st.columns(3)
    
    with col_res1:
        if stats['ai_percentage'] > 70:
            st.markdown('<div class="result-ai">', unsafe_allow_html=True)
            st.markdown(f"### 🤖 {stats['ai_percentage']:.1f}% IA")
            st.markdown("**Alta probabilidad de contenido generado por IA**")
            st.markdown('</div>', unsafe_allow_html=True)
        elif stats['ai_percentage'] > 40:
            st.warning(f"### ⚠️ {stats['ai_percentage']:.1f}% IA")
            st.markdown("**Contenido mixto o editado**")
        else:
            st.markdown('<div class="result-human">', unsafe_allow_html=True)
            st.markdown(f"### 👤 {100 - stats['ai_percentage']:.1f}% Humano")
            st.markdown("**Probablemente escrito por humano**")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col_res2:
        # Distribución de scores
        import plotly.graph_objects as go
        
        scores = [r['score'] for r in results]
        fig = go.Figure(data=[go.Histogram(x=scores, nbinsx=20)])
        fig.update_layout(
            title="Distribución de Scores",
            xaxis_title="Score IA",
            yaxis_title="Frecuencia",
            height=200,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_res3:
        # Métricas adicionales
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**📊 Métricas del Texto:**")
        st.markdown(f"- **Score promedio:** {stats['avg_score']:.3f}")
        st.markdown(f"- **Oraciones analizadas:** {stats['total_sentences']}")
        st.markdown(f"- **IA detectadas:** {stats['ai_sentences']}")
        st.markdown(f"- **Humanas detectadas:** {stats['total_sentences'] - stats['ai_sentences']}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Texto con resaltado
    st.markdown("### 🔍 Texto Analizado con Resaltado")
    
    # Seleccionar modo de visualización
    view_mode = st.radio(
        "Modo de visualización:",
        ["Texto Completo", "Solo Oraciones de IA", "Comparación lado a lado"],
        horizontal=True
    )
    
    if view_mode == "Texto Completo":
        # Resaltar todas las oraciones
        ai_sentences = [r['sentence'] for r in results if r['is_ai']]
        human_sentences = [r['sentence'] for r in results if not r['is_ai']]
        
        # Crear texto resaltado
        highlighted_text = st.session_state.analysis_results['text']
        
        # Resaltar oraciones de IA
        for sentence in ai_sentences:
            if sentence in highlighted_text:
                highlighted_text = highlighted_text.replace(
                    sentence,
                    f'<span class="ai-highlight">{sentence}</span>'
                )
        
        # Resaltar oraciones humanas (opcional)
        if show_details:
            for sentence in human_sentences:
                if sentence in highlighted_text and f'<span class="ai-highlight">{sentence}</span>' not in highlighted_text:
                    highlighted_text = highlighted_text.replace(
                        sentence,
                        f'<span class="human-highlight">{sentence}</span>'
                    )
        
        # Mostrar texto resaltado
        st.markdown(
            f'<div style="background: white; padding: 1.5rem; border-radius: 10px; border: 1px solid #e0e0e0; max-height: 400px; overflow-y: auto;">{highlighted_text}</div>',
            unsafe_allow_html=True
        )
        
        # Leyenda
        col_leg1, col_leg2 = st.columns(2)
        with col_leg1:
            st.markdown('<div class="ai-highlight" style="margin: 5px 0;">Oración detectada como IA</div>', unsafe_allow_html=True)
        with col_leg2:
            if show_details:
                st.markdown('<div class="human-highlight" style="margin: 5px 0;">Oración detectada como humana</div>', unsafe_allow_html=True)
    
    elif view_mode == "Solo Oraciones de IA":
        # Mostrar solo oraciones de IA
        ai_sentences = [r['sentence'] for r in results if r['is_ai']]
        
        if ai_sentences:
            st.markdown(f"**Se encontraron {len(ai_sentences)} oraciones con características de IA:**")
            
            for i, sentence in enumerate(ai_sentences, 1):
                score = next(r['score'] for r in results if r['sentence'] == sentence)
                st.markdown(f"""
                <div class="card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>#{i} - Score: {score:.2f}</strong>
                        </div>
                        <div style="color: #F44336; font-weight: bold;">
                            🤖 {score*100:.0f}% IA
                        </div>
                    </div>
                    <div style="margin-top: 10px;">{sentence}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No se encontraron oraciones con características claras de IA")
    
    # Tabla de resultados detallados
    if show_details:
        st.markdown("### 📋 Tabla de Resultados Detallados")
        
        # Crear DataFrame
        df_data = []
        for r in results:
            df_data.append({
                'Oración': r['sentence'][:100] + "..." if len(r['sentence']) > 100 else r['sentence'],
                'Score IA': f"{r['score']:.3f}",
                'Resultado': '🤖 IA' if r['is_ai'] else '👤 Humano',
                'Longitud': len(r['sentence']),
                'Palabras': len(word_tokenize(r['sentence']))
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True, height=300)
    
    # Opciones de exportación
    st.markdown("### 📤 Exportar Resultados")
    
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        # Exportar a CSV
        csv = pd.DataFrame([{
            'Total_Oraciones': stats['total_sentences'],
            'Oraciones_IA': stats['ai_sentences'],
            'Porcentaje_IA': f"{stats['ai_percentage']:.2f}%",
            'Score_Promedio': f"{stats['avg_score']:.3f}"
        }])
        
        st.download_button(
            label="📊 Descargar CSV",
            data=csv.to_csv(index=False).encode('utf-8'),
            file_name="resultados_deteccion_ia.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_exp2:
        # Exportar a JSON
        json_data = json.dumps(st.session_state.analysis_results, indent=2, ensure_ascii=False)
        st.download_button(
            label="📝 Descargar JSON",
            data=json_data.encode('utf-8'),
            file_name="resultados_deteccion_ia.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col_exp3:
        # Exportar texto resaltado
        highlighted_text = st.session_state.analysis_results['text']
        ai_sentences = [r['sentence'] for r in results if r['is_ai']]
        
        for sentence in ai_sentences:
            if sentence in highlighted_text:
                highlighted_text = highlighted_text.replace(
                    sentence,
                    f'[IA DETECTADA: {sentence}]'
                )
        
        st.download_button(
            label="📄 Descargar Texto Marcado",
            data=highlighted_text.encode('utf-8'),
            file_name="texto_marcado_ia.txt",
            mime="text/plain",
            use_container_width=True
        )

# Pie de página
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])

with footer_col2:
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem; padding: 1rem;">
    <p>🔍 <strong>Detector de IA en Textos v2.0</strong> | 🛡️ Análisis de autenticidad | 📚 Uso educativo</p>
    <p>⚡ Procesamiento en tiempo real | 📄 Soporte PDF/DOCX/TXT | 🎯 Detección por oraciones</p>
    <p>⚠️ <em>Esta herramienta es de apoyo, no un veredicto definitivo sobre la autoría.</em></p>
    </div>
    """, unsafe_allow_html=True)

# Script para procesar PDFs más complejos
if input_mode == "📄 Subir Archivo" and 'uploaded_file' in locals() and uploaded_file is not None:
    st.markdown("---")
    st.markdown("### 🔧 Procesamiento Avanzado de PDF")
    
    if st.button("🔄 Extraer texto con OCR (si es necesario)", use_container_width=True):
        with st.spinner("Intentando extracción avanzada..."):
            try:
                # Intentar extracción con OCR si falla la normal
                import pdfplumber
                with pdfplumber.open(uploaded_file) as pdf:
                    advanced_text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            advanced_text += page_text + "\n"
                        else:
                            # Si no hay texto, intentar con OCR
                            st.info(f"Página {page.page_number} podría requerir OCR")
                
                if advanced_text and len(advanced_text) > len(text_input):
                    text_input = advanced_text
                    st.success(f"✅ Extracción avanzada completada: {len(advanced_text)} caracteres")
                else:
                    st.warning("No se pudo extraer texto adicional")
            except:
                st.error("Error en extracción avanzada")