import streamlit as st
import PyPDF2
from transformers import pipeline
import torch
import re
from io import BytesIO

# Configuración de la página
st.set_page_config(
    page_title="Detector de Texto con IA",
    page_icon="🔍",
    layout="wide"
)

# Cachear el modelo para no cargarlo cada vez
@st.cache_resource
def load_model():
    try:
        # Usamos un modelo de clasificación de texto
        # Alternativas: "roberta-base-openai-detector" o "Hello-SimpleAI/chatgpt-detector-roberta"
        classifier = pipeline(
            "text-classification",
            model="Hello-SimpleAI/chatgpt-detector-roberta",
            device=-1  # CPU
        )
        return classifier
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        return None

def extract_text_from_pdf(pdf_file):
    """Extrae texto de un archivo PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extrayendo texto del PDF: {e}")
        return None

def split_text_into_chunks(text, chunk_size=400):
    """Divide el texto en fragmentos para análisis"""
    # Dividir por párrafos primero
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) < chunk_size:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return [chunk for chunk in chunks if len(chunk.strip()) > 50]

def analyze_text(text, classifier):
    """Analiza el texto y devuelve probabilidades"""
    if not text or len(text.strip()) < 50:
        return None
    
    try:
        # Limitar el texto a 512 tokens (aproximadamente 400 caracteres para estar seguros)
        # El modelo RoBERTa tiene un límite estricto de 512 tokens
        text_sample = text[:400]
        result = classifier(text_sample, truncation=True, max_length=512)[0]
        
        # El modelo devuelve 'LABEL_0' para humano y 'LABEL_1' para IA
        if result['label'] == 'LABEL_1':
            ai_probability = result['score']
        else:
            ai_probability = 1 - result['score']
        
        return ai_probability
    except Exception as e:
        st.warning(f"Error analizando fragmento: {e}")
        return None

# Título y descripción
st.title("🔍 Detector de Texto Generado por IA")
st.markdown("""
Esta aplicación analiza documentos PDF para detectar si el contenido ha sido generado por Inteligencia Artificial.
Sube un PDF y obtén un análisis detallado.
""")

# Sidebar con información
with st.sidebar:
    st.header("ℹ️ Información")
    st.markdown("""
    **¿Cómo funciona?**
    
    1. Sube tu archivo PDF
    2. El sistema extrae el texto
    3. Analiza cada sección con un modelo de ML
    4. Muestra resultados detallados
    
    **Nota:** Los resultados son estimaciones basadas en patrones de texto.
    """)
    
    st.markdown("---")
    st.markdown("**Modelo utilizado:**")
    st.info("ChatGPT Detector (RoBERTa)")

# Cargar el modelo
with st.spinner("Cargando modelo de detección..."):
    classifier = load_model()

if classifier is None:
    st.error("No se pudo cargar el modelo. Por favor, recarga la página.")
    st.stop()

# Upload de archivo
uploaded_file = st.file_uploader("Sube tu archivo PDF", type=['pdf'])

if uploaded_file is not None:
    # Extraer texto
    with st.spinner("Extrayendo texto del PDF..."):
        text = extract_text_from_pdf(uploaded_file)
    
    if text:
        # Mostrar estadísticas básicas
        st.success(f"✅ Texto extraído: {len(text)} caracteres, ~{len(text.split())} palabras")
        
        # Dividir en fragmentos
        with st.spinner("Analizando contenido..."):
            chunks = split_text_into_chunks(text, chunk_size=400)
            
            if len(chunks) == 0:
                st.warning("El documento no contiene suficiente texto para analizar.")
                st.stop()
            
            st.info(f"Analizando {len(chunks)} fragmentos del documento...")
            
            # Analizar cada fragmento
            results = []
            progress_bar = st.progress(0)
            
            for i, chunk in enumerate(chunks):
                prob = analyze_text(chunk, classifier)
                if prob is not None:
                    results.append({
                        'chunk': chunk,
                        'ai_probability': prob
                    })
                progress_bar.progress((i + 1) / len(chunks))
            
            progress_bar.empty()
        
        if results:
            # Calcular probabilidad promedio
            avg_probability = sum(r['ai_probability'] for r in results) / len(results)
            
            # Mostrar resultado general
            st.markdown("---")
            st.header("📊 Resultado General")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Probabilidad de IA",
                    f"{avg_probability * 100:.1f}%"
                )
            
            with col2:
                if avg_probability > 0.7:
                    verdict = "🤖 Probablemente IA"
                    color = "red"
                elif avg_probability > 0.4:
                    verdict = "⚠️ Mixto/Incierto"
                    color = "orange"
                else:
                    verdict = "✍️ Probablemente Humano"
                    color = "green"
                st.metric("Veredicto", verdict)
            
            with col3:
                st.metric("Fragmentos analizados", len(results))
            
            # Gráfico de distribución
            st.markdown("---")
            st.subheader("📈 Distribución de Probabilidades")
            
            # Crear datos para el gráfico
            import pandas as pd
            df = pd.DataFrame({
                'Fragmento': [f"Fragmento {i+1}" for i in range(len(results))],
                'Probabilidad IA (%)': [r['ai_probability'] * 100 for r in results]
            })
            
            st.bar_chart(df.set_index('Fragmento'))
            
            # Mostrar fragmentos sospechosos
            st.markdown("---")
            st.subheader("🔍 Análisis Detallado por Fragmentos")
            
            # Ordenar por probabilidad descendente
            sorted_results = sorted(results, key=lambda x: x['ai_probability'], reverse=True)
            
            for i, result in enumerate(sorted_results[:10]):  # Mostrar top 10
                prob = result['ai_probability']
                
                if prob > 0.7:
                    color_box = "🔴"
                    label = "Alta probabilidad de IA"
                elif prob > 0.4:
                    color_box = "🟡"
                    label = "Probabilidad media"
                else:
                    color_box = "🟢"
                    label = "Baja probabilidad de IA"
                
                with st.expander(f"{color_box} Fragmento {i+1} - {label} ({prob*100:.1f}%)"):
                    st.markdown(f"**Probabilidad de IA:** {prob*100:.1f}%")
                    st.text_area(
                        "Contenido:",
                        result['chunk'][:500] + "..." if len(result['chunk']) > 500 else result['chunk'],
                        height=150,
                        key=f"chunk_{i}"
                    )
            
            if len(sorted_results) > 10:
                st.info(f"Mostrando los 10 fragmentos con mayor probabilidad. Total de fragmentos: {len(sorted_results)}")
        else:
            st.error("No se pudo analizar ningún fragmento del documento.")
    else:
        st.error("No se pudo extraer texto del PDF. Asegúrate de que el archivo no esté protegido o sea una imagen escaneada.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Esta herramienta proporciona estimaciones basadas en modelos de aprendizaje automático.</p>
    <p>Los resultados deben interpretarse como indicadores, no como pruebas definitivas.</p>
</div>
""", unsafe_allow_html=True)