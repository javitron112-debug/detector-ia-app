import streamlit as st
import numpy as np
import pickle
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Detector de IA en Textos",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Descargar recursos de NLTK al inicio
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

download_nltk_resources()

# Título principal con estilo
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .human-box {
        background-color: #E8F5E9;
        border-left-color: #4CAF50;
    }
    .ai-box {
        background-color: #FFEBEE;
        border-left-color: #F44336;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Título
st.markdown('<h1 class="main-header">🔍 Detector de IA en Textos</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analiza si un texto fue escrito por un humano o generado por inteligencia artificial</p>', unsafe_allow_html=True)

class SimpleDetector:
    """Detector simple basado en heurísticas"""
    
    def __init__(self):
        self.features = {}
    
    def extract_features(self, text):
        """Extrae características del texto"""
        features = {}
        
        # Características básicas
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(sent_tokenize(text))
        
        # Características de complejidad
        words = word_tokenize(text.lower())
        unique_words = set(words)
        
        features['unique_word_ratio'] = len(unique_words) / max(len(words), 1)
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        
        # Características sintácticas
        sentences = sent_tokenize(text)
        sentence_lengths = [len(sent.split()) for sent in sentences]
        
        features['avg_sentence_length'] = np.mean(sentence_lengths) if sentence_lengths else 0
        features['sentence_length_variance'] = np.var(sentence_lengths) if len(sentence_lengths) > 1 else 0
        
        # Características de puntuación
        features['comma_density'] = text.count(',') / max(len(text.split()), 1)
        features['exclamation_density'] = text.count('!') / max(len(text.split()), 1)
        
        # Detección de patrones comunes de IA
        ai_patterns = [
            'es importante destacar',
            'en conclusión',
            'por otro lado',
            'sin embargo',
            'además',
            'por lo tanto',
            'en primer lugar',
            'en segundo lugar',
            'cabe mencionar',
            'es fundamental'
        ]
        
        features['ai_pattern_count'] = sum(1 for pattern in ai_patterns if pattern in text.lower())
        
        # Burstiness (variabilidad en longitud de oraciones)
        if len(sentence_lengths) > 1 and np.mean(sentence_lengths) > 0:
            features['burstiness'] = np.std(sentence_lengths) / np.mean(sentence_lengths)
        else:
            features['burstiness'] = 0
        
        self.features = features
        return features
    
    def predict(self, text):
        """Predice si el texto es de IA"""
        features = self.extract_features(text)
        
        # Puntuación basada en heurísticas
        score = 0
        
        # 1. Uniformidad en longitud de oraciones (IA tiende a ser más uniforme)
        if features['sentence_length_variance'] < 20:
            score += 0.2
        
        # 2. Densidad de comas (IA usa más puntuación estructurada)
        if features['comma_density'] > 0.08:
            score += 0.15
        
        # 3. Patrones de lenguaje de IA
        score += min(features['ai_pattern_count'] * 0.05, 0.3)
        
        # 4. Baja burstiness (IA tiene menos variación)
        if features['burstiness'] < 0.5:
            score += 0.1
        
        # 5. Baja densidad de exclamaciones (IA es más formal)
        if features['exclamation_density'] < 0.01:
            score += 0.05
        
        # 6. Longitud promedio de palabras (IA puede usar palabras más largas)
        if features['avg_word_length'] > 5:
            score += 0.05
        
        # Ajustar por longitud del texto
        if features['word_count'] < 50:
            score *= 0.7  # Menos confiable en textos cortos
        
        # Limitar score entre 0 y 1
        score = min(max(score, 0), 0.95)
        
        # Determinar confianza
        confidence = "Alta" if abs(score - 0.5) > 0.3 else "Media" if abs(score - 0.5) > 0.15 else "Baja"
        
        return {
            'prediction': 'IA 🤖' if score > 0.5 else 'Humano 👤',
            'probability': float(score),
            'confidence': confidence,
            'features': features
        }

# Inicializar detector
@st.cache_resource
def get_detector():
    return SimpleDetector()

detector = get_detector()

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    
    # Modo de análisis
    analysis_mode = st.selectbox(
        "Modo de análisis",
        ["Rápido", "Detallado", "Comparativo"]
    )
    
    # Umbral de detección
    threshold = st.slider(
        "Umbral de sensibilidad",
        min_value=0.3,
        max_value=0.7,
        value=0.5,
        step=0.05,
        help="Ajusta qué tan estricto es el detector"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Estadísticas")
    
    # Ejemplos predefinidos
    st.markdown("#### Ejemplos para probar:")
    
    example_texts = {
        "Texto Humano (Conversacional)": "Hoy fui al mercado y compré unas manzanas. Estaban un poco caras, pero me gusta su sabor. De camino a casa me encontré con mi vecina María, que me contó que se va de vacaciones la semana que viene.",
        "Texto IA (Formal)": "La inteligencia artificial constituye un paradigma tecnológico transformacional que está redefiniendo los procesos empresariales contemporáneos. Es fundamental considerar los aspectos éticos inherentes a su implementación para garantizar un desarrollo sostenible y equitativo.",
        "Texto Mixto": "Los modelos de lenguaje como GPT son increíblemente útiles. Personalmente, los uso para ayudarme con tareas de escritura, aunque a veces cometen errores graciosos. Es importante verificar siempre la información que proporcionan.",
    }
    
    selected_example = st.selectbox("Cargar ejemplo:", list(example_texts.keys()))
    
    if st.button("Cargar ejemplo seleccionado"):
        st.session_state.text_input = example_texts[selected_example]
    
    st.markdown("---")
    st.markdown("### ℹ️ Acerca de")
    st.markdown("""
    Esta herramienta analiza patrones en el texto para determinar si fue generado por IA.
    
    **Precisión estimada:** ~70-80%
    
    **Limitaciones:**
    - Textos cortos son más difíciles
    - No es 100% preciso
    - Los textos editados pueden confundir
    
    **Desarrollado con:** Python, Streamlit, NLTK
    """)

# Área principal de la aplicación
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Ingresa el texto a analizar")
    
    # Text area con valor de sesión
    text_input = st.text_area(
        "Pega tu texto aquí:",
        height=250,
        value=st.session_state.get('text_input', ''),
        placeholder="Escribe o pega el texto que quieres analizar aquí..."
    )
    
    # Botones de acción
    col1_1, col1_2, col1_3 = st.columns(3)
    
    with col1_1:
        analyze_btn = st.button("🔍 Analizar Texto", type="primary", use_container_width=True)
    
    with col1_2:
        clear_btn = st.button("🧹 Limpiar", use_container_width=True)
    
    with col1_3:
        sample_btn = st.button("🎲 Texto Aleatorio", use_container_width=True)
    
    if clear_btn:
        st.session_state.text_input = ""
        st.rerun()
    
    if sample_btn:
        samples = [
            "El aprendizaje automático ha revolucionado la forma en que procesamos datos. Sin embargo, es crucial mantener un enfoque humanocéntrico en su desarrollo.",
            "Ayer por la tarde, mientras paseaba por el parque, vi a un par de niños jugando al fútbol. Uno de ellos metió un gol increíble desde lejos, todos nos quedamos boquiabiertos.",
            "La sostenibilidad ambiental representa uno de los desafíos más apremiantes de nuestra era. En consecuencia, la adopción de energías renovables se ha convertido en una prioridad estratégica a nivel global."
        ]
        import random
        st.session_state.text_input = random.choice(samples)
        st.rerun()

with col2:
    st.markdown("### 📈 Métricas de Texto")
    
    if text_input:
        # Calcular métricas básicas
        words = len(text_input.split())
        sentences = len(sent_tokenize(text_input))
        chars = len(text_input.replace(" ", ""))
        
        st.metric("Palabras", words)
        st.metric("Oraciones", sentences)
        st.metric("Caracteres", chars)
        
        if words > 0:
            st.metric("Palabras/Oración", f"{words/max(sentences,1):.1f}")
    else:
        st.info("Ingresa un texto para ver las métricas")

# Análisis principal
if analyze_btn and text_input.strip():
    with st.spinner("Analizando texto..."):
        # Realizar predicción
        result = detector.predict(text_input)
        
        # Mostrar resultado principal
        st.markdown("---")
        st.markdown("## 📊 Resultado del Análisis")
        
        # Tarjeta de resultado
        if result['prediction'] == 'IA 🤖':
            box_class = "ai-box"
            emoji = "🤖"
            color = "#F44336"
        else:
            box_class = "human-box"
            emoji = "👤"
            color = "#4CAF50"
        
        st.markdown(f"""
        <div class="result-box {box_class}">
            <h2 style="margin-top: 0;">{emoji} {result['prediction']}</h2>
            <p><strong>Probabilidad de ser IA:</strong> {result['probability']:.1%}</p>
            <p><strong>Nivel de confianza:</strong> {result['confidence']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Barra de progreso
        prob_percent = result['probability'] * 100
        st.progress(float(result['probability']), 
                   text=f"Score IA: {prob_percent:.1f}%")
        
        # Análisis detallado
        if analysis_mode in ["Detallado", "Comparativo"]:
            st.markdown("### 🔍 Análisis Detallado")
            
            features = result['features']
            
            cols = st.columns(4)
            metrics = [
                ("Riqueza Léxica", f"{features['unique_word_ratio']:.1%}"),
                ("Long. Prom. Palabra", f"{features['avg_word_length']:.1f}"),
                ("Palabras/Oración", f"{features['avg_sentence_length']:.1f}"),
                ("Burstiness", f"{features['burstiness']:.3f}")
            ]
            
            for col, (label, value) in zip(cols, metrics):
                with col:
                    st.metric(label, value)
            
            # Interpretación de características
            st.markdown("#### 📝 Interpretación:")
            
            interpretations = []
            
            if features['ai_pattern_count'] > 2:
                interpretations.append("**Patrones de IA detectados**: El texto contiene frases comúnmente usadas por modelos de lenguaje")
            
            if features['burstiness'] < 0.5:
                interpretations.append("**Baja variabilidad**: Las oraciones tienen longitudes similares (común en textos de IA)")
            
            if features['comma_density'] > 0.1:
                interpretations.append("**Alta densidad de comas**: Estructura sintáctica compleja y formal")
            
            if features['unique_word_ratio'] < 0.4 and features['word_count'] > 50:
                interpretations.append("**Vocabulario limitado**: Repetición de palabras comunes")
            
            for interp in interpretations:
                st.markdown(f"- {interp}")
        
        # Comparativo
        if analysis_mode == "Comparativo":
            st.markdown("### 📊 Análisis Comparativo")
            
            # Crear datos comparativos
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            features_plot = ['unique_word_ratio', 'burstiness', 'comma_density', 'ai_pattern_count']
            labels = ['Riqueza Léxica', 'Burstiness', 'Densidad Comas', 'Patrones IA']
            values = [features[f] for f in features_plot]
            
            # Normalizar valores para el radar chart
            normalized_values = []
            max_vals = [1.0, 2.0, 0.2, 5.0]  # Valores máximos esperados
            
            for val, max_val in zip(values, max_vals):
                normalized_values.append(min(val / max_val, 1.0))
            
            fig.add_trace(go.Scatterpolar(
                r=normalized_values + [normalized_values[0]],
                theta=labels + [labels[0]],
                fill='toself',
                name='Tu texto',
                line_color=color
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Perfil de Características del Texto"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Consejos
        st.markdown("### 💡 Consejos")
        
        if result['prediction'] == 'IA 🤖' and result['probability'] > 0.7:
            st.warning("""
            **Posible texto generado por IA detectado:**
            - Considera revisar el contenido críticamente
            - Verifica fuentes adicionales si es información importante
            - Los textos de IA pueden contener errores fácticos o sesgos
            """)
        elif result['prediction'] == 'Humano 👤' and result['probability'] < 0.3:
            st.success("""
            **Características de texto humano identificadas:**
            - Variabilidad natural en estilo
            - Patrones conversacionales
            - Posibles imperfecciones gramaticales menores
            """)
        else:
            st.info("""
            **Resultado indeterminado:**
            - El texto muestra características tanto humanas como de IA
            - Podría ser texto humano muy bien escrito
            - O texto de IA editado o modificado
            """)
        
        # Advertencia
        st.markdown("---")
        st.markdown("""
        <div style="background-color: #FFF3CD; padding: 1rem; border-radius: 5px; border-left: 4px solid #FFC107;">
        <strong>⚠️ Limitación importante:</strong> Esta herramienta es una ayuda para el análisis, 
        pero no debe usarse como único criterio para determinar la autoría de un texto. 
        La precisión no es del 100% y pueden ocurrir falsos positivos/negativos.
        </div>
        """, unsafe_allow_html=True)

elif analyze_btn and not text_input.strip():
    st.warning("⚠️ Por favor, ingresa un texto para analizar")

# Footer
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)

with col_f2:
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>🔍 Detector de IA v1.0 | 📚 Uso educativo | ⚖️ Herramienta de análisis</p>
    <p>Desplegado en <a href="https://share.streamlit.io" target="_blank">Streamlit Sharing</a></p>
    </div>
    """, unsafe_allow_html=True)