import os
import streamlit as st
import pandas as pd

from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent

# Configuración de la página
st.set_page_config(page_title="Gym & Data AI 🏋️‍♂️", page_icon="📊")
st.title("Habla con tu Excel (Coach de Gimnasio 🤖)")

# -------------------------------
# 🔑 API KEY
# -------------------------------
st.sidebar.header("🔐 Configuración")
api_key_input = st.sidebar.text_input(
    "Introduce tu Groq API Key:",
    type="password",
    placeholder="gsk-xxxxxxxxxxxxxxxx"
)

api_key = api_key_input or os.getenv("GROQ_API_KEY")

if not api_key:
    st.warning("⚠️ Debes introducir tu Groq API Key para continuar.")
    st.stop()

# -------------------------------
# 📄 Carga de Datos
# -------------------------------
EXCEL_PATH = "Empleados.arff.csv.xlsx" # Cambia esto por tu archivo de gym cuando quieras

if not os.path.exists(EXCEL_PATH):
    st.error(f"No encuentro el archivo '{EXCEL_PATH}'.")
    st.stop()

try:
    # Usamos openpyxl como motor para asegurar compatibilidad con .xlsx
    df = pd.read_excel(EXCEL_PATH, engine='openpyxl')
except Exception as e:
    st.error(f"No se pudo leer el Excel: {e}")
    st.stop()

# -------------------------------
# 👀 Vista previa
# -------------------------------
with st.expander("👀 Ver muestra de los datos"):
    st.write(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
    st.dataframe(df.head(10), use_container_width=True)

st.divider()

# -------------------------------
# 💬 Consultas
# -------------------------------
st.subheader("Haz tu consulta técnica o de entrenamiento")
question = st.text_input("Ejemplo: ¿Cuál es el promedio de la columna X o mi mejor marca?")

if st.button("Consultar 🚀"):
    if not question.strip():
        st.warning("Escribe una pregunta primero.")
    else:
        with st.spinner("La IA está analizando los datos..."):
            try:
                # 1. Inicialización correcta del LLM
                # Usamos 'model' y pasamos la API KEY validada
                llm = ChatGroq(
                    api_key=api_key, 
                    model="llama-3.3-70b-versatile",
                    temperature=0.1 # Menor temperatura = más precisión matemática
                )

                # 2. Creación del Agente corregida
                # El agent_type "tool-calling" evita el error de NoneType
                agent = create_pandas_dataframe_agent(
                    llm,
                    df,
                    verbose=False,
                    allow_dangerous_code=True,
                    agent_type="tool-calling", 
                )

                # 3. Ejecución con estructura de diccionario para mayor estabilidad
                response = agent.invoke({"input": question})
                
                st.success("Análisis completado:")
                
                # Extraer la respuesta correctamente
                if isinstance(response, dict) and "output" in response:
                    st.markdown(response["output"])
                else:
                    st.write(response)

            except Exception as e:
                # Captura de errores específica para guiar al usuario
                error_msg = str(e)
                if "tabulate" in error_msg.lower():
                    st.error("Error: Falta la librería 'tabulate'. Añádela a tu requirements.txt.")
                else:
                    st.error(f"Error consultando el Excel: {e}")

# Pie de página opcional
st.caption("Desarrollado con Streamlit + LangChain + Groq")

