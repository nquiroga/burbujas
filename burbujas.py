import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import math
import random

# Configuración de la página
st.set_page_config(page_title="Análisis de Burbujas - Palabras en Discursos", layout="wide")

# Título de la aplicación
st.title("📊 Análisis de Burbujas: Comparación de Palabras entre Años")

# Cargar datos (con cache para mejorar rendimiento)
@st.cache_data
def cargar_datos():
    df = pd.read_csv('discursos_presidenciales_limpios.csv')
    df['anio'] = df['anio'].astype(int)
    return df

class NYTBubbleControlManual:
    """
    Versión con control manual de palabras
    """

    def __init__(self, figsize=(20, 12)):
        self.figsize = figsize
        self.width = 1000
        self.height = 600
        self.min_radius = 20
        self.max_radius = 80
        self.collision_padding = 10

        # Colores NYT
        self.color_party1 = '#4A90E2'  # Azul
        self.color_party2 = '#E94B3C'  # Rojo

    def process_data(self, df, anios_seleccionados, n_words=25,
                     palabras_excluir=None, palabras_manuales=None):
        """
        Procesa los datos con control manual de palabras
        """

        speeches = df[df['anio'].isin(anios_seleccionados)].copy()

        if len(speeches) != 2:
            raise ValueError(f"Se necesitan exactamente 2 discursos, se encontraron {len(speeches)}")

        speeches = speeches.sort_values('anio').reset_index(drop=True)

        self.party1_name = speeches.iloc[0]['presidente']
        self.party2_name = speeches.iloc[1]['presidente']
        self.year1 = speeches.iloc[0]['anio']
        self.year2 = speeches.iloc[1]['anio']

        text1 = speeches.iloc[0]['texto_limpio']
        text2 = speeches.iloc[1]['texto_limpio']

        # Calcular word counts
        words1 = len(text1.split())
        words2 = len(text2.split())

        # Convertir listas a sets para búsqueda más rápida
        palabras_excluir = set(palabras_excluir or [])

        # Decidir qué palabras analizar
        if palabras_manuales:
            # Modo manual: usar lista predefinida
            selected_words = list(palabras_manuales)
        else:
            # Modo automático: usar TF-IDF con exclusiones
            selected_words = self.extract_words_tfidf(text1, text2, n_words, palabras_excluir)

        # Crear topics
        self.topics = []
        for word in selected_words:
            topic = self.create_topic(word, text1, text2, words1, words2)
            if topic['count_total'] > 0:
                self.topics.append(topic)

        return self.topics

    def extract_words_tfidf(self, text1, text2, n_words, palabras_excluir):
        """Extrae palabras usando TF-IDF con exclusiones"""

        # Crear vocabulario personalizado excluyendo palabras no deseadas
        def tokenizer_personalizado(text):
            tokens = re.findall(r'\b\w{3,}\b', text.lower())
            return [token for token in tokens if token not in palabras_excluir]

        vectorizer = TfidfVectorizer(
            max_features=200,
            min_df=1,
            tokenizer=tokenizer_personalizado,
            lowercase=False
        )

        try:
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            feature_names = vectorizer.get_feature_names_out()

            # Calcular importancia total
            tfidf_total = np.array(tfidf_matrix.sum(axis=0)).flatten()
            top_indices = np.argsort(tfidf_total)[-n_words:][::-1]
            selected_words = [feature_names[i] for i in top_indices]

            return selected_words
        except ValueError as e:
            return self.extract_words_frequency(text1, text2, n_words, palabras_excluir)

    def extract_words_frequency(self, text1, text2, n_words, palabras_excluir):
        """Fallback: extraer por frecuencia si TF-IDF falla"""

        from collections import Counter

        # Combinar textos y contar
        all_words = re.findall(r'\b\w{3,}\b', (text1 + ' ' + text2).lower())
        all_words = [w for w in all_words if w not in palabras_excluir]

        word_counts = Counter(all_words)
        return [word for word, count in word_counts.most_common(n_words)]

    def create_topic(self, word, text1, text2, words1, words2):
        """Crea un topic con conteos directos"""

        # Contar ocurrencias exactas (case-insensitive)
        pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
        count1 = len(pattern.findall(text1))
        count2 = len(pattern.findall(text2))

        # Normalizar por 25,000 palabras como NYT
        norm_count1 = (count1 / words1) * 25000 if words1 > 0 else 0
        norm_count2 = (count2 / words2) * 25000 if words2 > 0 else 0

        total_count = norm_count1 + norm_count2

        # Calcular fracción k
        k = norm_count1 / total_count if total_count > 0 else 0.5

        return {
            'name': word,
            'count_1': norm_count1,
            'count_2': norm_count2,
            'count_total': total_count,
            'raw_count_1': count1,
            'raw_count_2': count2,
            'k': k,
            'x': 0,
            'y': 0,
            'r': 0
        }

    def calculate_layout(self):
        """Calcula posiciones con mejor distribución espacial"""

        if not self.topics:
            return

        # Calcular radios
        max_count = max(t['count_total'] for t in self.topics)
        for topic in self.topics:
            if max_count > 0:
                normalized = math.sqrt(topic['count_total'] / max_count)
                topic['r'] = self.min_radius + normalized * (self.max_radius - self.min_radius)
            else:
                topic['r'] = self.min_radius

        # Ordenar por tamaño para mejor colocación
        self.topics.sort(key=lambda t: t['count_total'], reverse=True)

        # Distribución inicial mejorada
        margin = 100
        usable_width = self.width - 2 * margin
        usable_height = self.height - 2 * margin

        for i, topic in enumerate(self.topics):
            # Posición X basada en k (sesgo político)
            bias_x = topic['k']  # 0 = todo derecha, 1 = todo izquierda
            base_x = margin + (1 - bias_x) * usable_width

            # Añadir variación para evitar superposición vertical
            variation = random.uniform(-50, 50)
            topic['x'] = max(margin + topic['r'],
                           min(self.width - margin - topic['r'], base_x + variation))

            # Distribución Y en capas
            rows = 4
            row = i % rows
            y_base = margin + (row + 0.5) * (usable_height / rows)
            y_variation = random.uniform(-30, 30)
            topic['y'] = max(margin + topic['r'],
                           min(self.height - margin - topic['r'], y_base + y_variation))

        # Resolver colisiones iterativamente
        self.resolve_collisions(iterations=100)

    def resolve_collisions(self, iterations=100):
        """Resuelve colisiones entre burbujas"""

        for iteration in range(iterations):
            moved = False

            for i, topic1 in enumerate(self.topics):
                for j, topic2 in enumerate(self.topics[i+1:], i+1):
                    dx = topic2['x'] - topic1['x']
                    dy = topic2['y'] - topic1['y']
                    distance = math.sqrt(dx*dx + dy*dy)
                    min_distance = topic1['r'] + topic2['r'] + self.collision_padding

                    if distance < min_distance and distance > 0:
                        # Calcular separación
                        overlap = min_distance - distance
                        dx_norm = dx / distance
                        dy_norm = dy / distance

                        # Mover proporcionalmente al tamaño
                        total_r = topic1['r'] + topic2['r']
                        move1 = overlap * (topic2['r'] / total_r) * 0.5
                        move2 = overlap * (topic1['r'] / total_r) * 0.5

                        topic1['x'] -= dx_norm * move1
                        topic1['y'] -= dy_norm * move1
                        topic2['x'] += dx_norm * move2
                        topic2['y'] += dy_norm * move2

                        moved = True

            # Mantener dentro de límites
            margin = 50
            for topic in self.topics:
                topic['x'] = max(margin + topic['r'],
                               min(self.width - margin - topic['r'], topic['x']))
                topic['y'] = max(margin + topic['r'],
                               min(self.height - margin - topic['r'], topic['y']))

            if not moved:
                break

    def create_visualization(self):
        """Crea la visualización"""

        plt.close('all')  # Limpiar figuras anteriores
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 150

        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        ax.set_facecolor('white')

        # Escalar coordenadas
        scale_x = 1.0 / self.width
        scale_y = 1.0 / self.height
        max_r = max(t['r'] for t in self.topics) if self.topics else 1
        scale_r = 0.05 / max_r  # Burbujas más pequeñas

        for topic in self.topics:
            x = topic['x'] * scale_x
            y = (self.height - topic['y']) * scale_y
            r = topic['r'] * scale_r
            k = topic['k']

            # Círculo base (partido 2)
            circle_base = Circle((x, y), r,
                               facecolor=self.color_party2,
                               edgecolor='black',
                               linewidth=1.5,
                               alpha=0.8)
            ax.add_patch(circle_base)

            # Overlay del partido 1 si k > 0
            if k > 0:
                if k >= 1:
                    # Todo el círculo
                    circle_overlay = Circle((x, y), r,
                                          facecolor=self.color_party1,
                                          edgecolor='black',
                                          linewidth=1.5,
                                          alpha=0.9)
                    ax.add_patch(circle_overlay)
                else:
                    # Usar wedge para crear sector
                    angle_degrees = 360 * k
                    start_angle = 90 - angle_degrees/2
                    end_angle = 90 + angle_degrees/2

                    wedge = Wedge((x, y), r, start_angle, end_angle,
                                facecolor=self.color_party1,
                                alpha=0.9)
                    ax.add_patch(wedge)

                    # Líneas de división
                    if 0.1 < k < 0.9:
                        angle1_rad = math.radians(start_angle)
                        angle2_rad = math.radians(end_angle)

                        x1 = x + r * math.cos(angle1_rad) * 0.8
                        y1 = y + r * math.sin(angle1_rad) * 0.8
                        x2 = x + r * math.cos(angle2_rad) * 0.8
                        y2 = y + r * math.sin(angle2_rad) * 0.8

                        ax.plot([x, x1], [y, y1], 'k-', alpha=0.3, linewidth=1)
                        ax.plot([x, x2], [y, y2], 'k-', alpha=0.3, linewidth=1)

            # Etiqueta de palabra arriba
            label_y = y + r + 0.02
            ax.text(x, label_y, topic['name'],
                   ha='center', va='bottom',
                   fontsize=max(10, min(16, r * 150)),
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

            # Conteos en el centro
            count1 = int(round(topic['count_1']))
            count2 = int(round(topic['count_2']))
            ax.text(x, y, f"{count1}-{count2}",
                   ha='center', va='center',
                   fontsize=max(8, min(12, r * 120)),
                   fontweight='bold',
                   color='white',
                   bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.6))

        # Configurar ejes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')

        # Título principal
        ax.text(0.5, 0.96, "Análisis de Discursos: Palabras Más Utilizadas",
               ha='center', va='top', fontsize=20, fontweight='bold',
               transform=ax.transAxes)

        ax.text(0.5, 0.93, f"{self.party1_name} ({self.year1}) vs {self.party2_name} ({self.year2})",
               ha='center', va='top', fontsize=14, style='italic',
               transform=ax.transAxes)

        # Leyendas laterales
        ax.text(0.15, 0.88, f"Palabras más usadas\npor {self.party1_name}",
               ha='center', va='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor=self.color_party1, alpha=0.8),
               transform=ax.transAxes, color='white')

        ax.text(0.85, 0.88, f"Palabras más usadas\npor {self.party2_name}",
               ha='center', va='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor=self.color_party2, alpha=0.8),
               transform=ax.transAxes, color='white')

        # Explicación inferior
        ax.text(0.5, 0.05, "Frecuencias normalizadas por 25,000 palabras | Tamaño = frecuencia total | Color = distribución entre presidentes",
               ha='center', va='center', fontsize=11, style='italic',
               transform=ax.transAxes)

        # Flechas indicativas
        ax.annotate('', xy=(0.3, 0.1), xytext=(0.2, 0.1),
                   arrowprops=dict(arrowstyle='<-', color='gray', lw=2),
                   transform=ax.transAxes)
        ax.text(0.15, 0.1, f'← {self.party1_name}', ha='center', va='center',
               fontsize=10, transform=ax.transAxes)

        ax.annotate('', xy=(0.8, 0.1), xytext=(0.7, 0.1),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=2),
                   transform=ax.transAxes)
        ax.text(0.85, 0.1, f'{self.party2_name} →', ha='center', va='center',
               fontsize=10, transform=ax.transAxes)

        plt.tight_layout()
        return fig

def analizar_discursos_manual(df, anios_seleccionados, palabras_manuales, figsize=(16, 10)):
    """
    Función principal con control manual de palabras
    """
    
    analyzer = NYTBubbleControlManual(figsize=figsize)
    
    try:
        # Procesar datos con palabras manuales
        topics = analyzer.process_data(df, anios_seleccionados, 
                                       palabras_manuales=palabras_manuales)
        
        if not topics:
            return None, None
        
        # Calcular layout
        analyzer.calculate_layout()
        
        # Crear visualización
        fig = analyzer.create_visualization()
        
        # Crear DataFrame de resultados
        results_data = []
        for topic in topics:
            results_data.append({
                'palabra': topic['name'],
                'freq_normalizada_1': round(topic['count_1'], 1),
                'freq_normalizada_2': round(topic['count_2'], 1),
                'freq_total': round(topic['count_total'], 1),
                'menciones_raw_1': topic['raw_count_1'],
                'menciones_raw_2': topic['raw_count_2'],
                'fraccion_k': round(topic['k'], 3),
                'radio': round(topic['r'], 1)
            })
        
        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values('freq_total', ascending=False)
        
        return fig, results_df
        
    except Exception as e:
        raise e

def validar_palabras(palabras_usuario):
    """
    Valida y limpia las palabras ingresadas por el usuario
    """
    # Limpiar y validar entrada de palabras
    # Remover comas múltiples, espacios extra y entradas vacías
    palabras_limpio = re.sub(r',+', ',', palabras_usuario.strip())  # Comas múltiples
    palabras_limpio = re.sub(r',\s*,', ',', palabras_limpio)        # Comas con espacios
    palabras_limpio = palabras_limpio.strip(',')                    # Comas al inicio/final
    
    # Procesar palabras
    palabras_lista = []
    palabras_ignoradas = []
    
    for palabra in palabras_limpio.split(','):
        palabra = palabra.strip().lower()
        if palabra and len(palabra) >= 2:  # Mínimo 2 caracteres
            # Validar que solo contenga letras y espacios (incluye acentos)
            if palabra.replace(' ', '').replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n').isalpha():
                palabras_lista.append(palabra)
            else:
                palabras_ignoradas.append(palabra)
        elif palabra:  # Palabra muy corta
            palabras_ignoradas.append(palabra)
    
    # Remover duplicados manteniendo orden
    palabras_lista = list(dict.fromkeys(palabras_lista))
    
    return palabras_lista, palabras_ignoradas

# Interfaz de usuario
try:
    df = cargar_datos()
    
    # Obtener años disponibles
    anios_disponibles = sorted(df['anio'].unique())
    
    # Crear columnas para la interfaz
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Configuración del Análisis")
        
        # Seleccionar años
        st.write("**Selecciona dos años para comparar:**")
        anio1 = st.selectbox("Primer año:", anios_disponibles, index=0)
        anio2 = st.selectbox("Segundo año:", anios_disponibles, index=min(len(anios_disponibles)-1, 1))
        
        if anio1 == anio2:
            st.warning("⚠️ Debes seleccionar dos años diferentes")
        
        # Ingresar palabras
        st.write("**Palabras a analizar:**")
        palabras_usuario = st.text_area(
            "Ingresa hasta 15 palabras separadas por coma:",
            value="democracia, economía, justicia, educación",
            help="Ejemplo: democracia, economía, justicia, educación, salud",
            height=100
        )
        
        if st.button("🔍 Generar Análisis de Burbujas", type="primary"):
            try:
                if palabras_usuario.strip() and anio1 != anio2:
                    # Validar palabras
                    palabras_lista, palabras_ignoradas = validar_palabras(palabras_usuario)
                    
                    # Mostrar advertencias si hay palabras ignoradas
                    if palabras_ignoradas:
                        st.warning(f"⚠️ Palabras ignoradas: {', '.join(palabras_ignoradas)}")
                    
                    if len(palabras_lista) > 15:
                        st.error("⚠️ Máximo 15 palabras permitidas")
                        st.info(f"📝 Ingresaste {len(palabras_lista)} palabras válidas. Reduce la cantidad.")
                    elif len(palabras_lista) == 0:
                        st.error("⚠️ No se encontraron palabras válidas")
                        st.info("💡 Verifica que las palabras tengan al menos 2 caracteres y solo contengan letras")
                    else:
                        # Verificar que existan datos para ambos años
                        datos_anio1 = df[df['anio'] == anio1]
                        datos_anio2 = df[df['anio'] == anio2]
                        
                        if len(datos_anio1) == 0:
                            st.error(f"❌ No se encontraron datos para el año {anio1}")
                        elif len(datos_anio2) == 0:
                            st.error(f"❌ No se encontraron datos para el año {anio2}")
                        else:
                            st.session_state.anios_analizar = [anio1, anio2]
                            st.session_state.palabras_analizar = palabras_lista
                            st.session_state.mostrar_analisis = True
                            st.success(f"✅ Analizando {len(palabras_lista)} palabras entre {anio1} y {anio2}")
                            if len(palabras_lista) != len(palabras_usuario.split(',')):
                                st.info(f"📋 Palabras procesadas: {', '.join(palabras_lista)}")
                else:
                    if not palabras_usuario.strip():
                        st.error("⚠️ Debes ingresar al menos una palabra")
                    if anio1 == anio2:
                        st.error("⚠️ Debes seleccionar dos años diferentes")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state.mostrar_analisis = False
    
    # Área principal para mostrar el análisis
    with col2:
        if hasattr(st.session_state, 'mostrar_analisis') and st.session_state.mostrar_analisis:
            try:
                with st.spinner("🔄 Generando análisis de burbujas..."):
                    fig, datos_resultado = analizar_discursos_manual(
                        df, 
                        st.session_state.anios_analizar, 
                        st.session_state.palabras_analizar
                    )
                    
                    if fig is None:
                        st.warning("⚠️ No se pudieron procesar las palabras solicitadas")
                        st.info("💡 Verifica que las palabras estén escritas correctamente")
                    else:
                        # Mostrar gráfico
                        st.pyplot(fig)
                        
                        # Mostrar tabla de resultados
                        st.subheader("📊 Resultados del Análisis")
                        
                        # Información de los presidentes
                        presidente1 = df[df['anio'] == st.session_state.anios_analizar[0]]['presidente'].iloc[0]
                        presidente2 = df[df['anio'] == st.session_state.anios_analizar[1]]['presidente'].iloc[0]
                        
                        col2a, col2b = st.columns(2)
                        with col2a:
                            st.metric("Año 1", f"{st.session_state.anios_analizar[0]} - {presidente1}")
                        with col2b:
                            st.metric("Año 2", f"{st.session_state.anios_analizar[1]} - {presidente2}")
                        
                        # Tabla de datos
                        st.dataframe(
                            datos_resultado[['palabra', 'freq_total', 'freq_normalizada_1', 'freq_normalizada_2', 'menciones_raw_1', 'menciones_raw_2']],
                            use_container_width=True
                        )
                        
            except Exception as e:
                st.error(f"❌ Error en el análisis: {str(e)}")
                st.session_state.mostrar_analisis = False

except FileNotFoundError:
    st.error("❌ No se encontró el archivo 'discursos_presidenciales_limpios.csv'")
    st.info("📁 Asegúrate de que el archivo esté en el mismo directorio que la aplicación")
except Exception as e:
    st.error(f"❌ Error al cargar los datos: {str(e)}")

# Información adicional
with st.expander("ℹ️ Cómo interpretar el análisis de burbujas"):
    st.write("""
    **📈 Elementos del gráfico:**
    - **Tamaño de burbuja**: Frecuencia total de la palabra (más grande = más mencionada)
    - **Color azul**: Palabras más usadas por el primer presidente
    - **Color rojo**: Palabras más usadas por el segundo presidente
    - **Números en burbujas**: Frecuencias normalizadas por 25,000 palabras
    
    **🎯 Posicionamiento:**
    - **Izquierda**: Palabras favorecidas por el primer año/presidente
    - **Derecha**: Palabras favorecidas por el segundo año/presidente
    - **Centro**: Palabras usadas de manera similar por ambos
    
    **💡 Tips de uso:**
    - Usa palabras relacionadas temáticamente: `democracia, libertad, justicia`
    - Prueba conceptos económicos: `empleo, inflación, crecimiento`
    - Analiza temas sociales: `educación, salud, seguridad`
    - Puedes ingresar hasta 15 palabras para análisis más completos
    - Las palabras duplicadas se eliminan automáticamente
    - Se ignoran palabras con caracteres especiales o muy cortas
    """)
