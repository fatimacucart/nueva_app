import os
import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentType

st.set_page_config(page_title="Excel AI 🤖", page_icon="📊")
st.title("💬📂 Habla con tu Excel gracias a la IA")

# --- SIDEBAR: API KEY ---
st.sidebar.header("🔐 Configuración")
api_key_input = st.sidebar.text_input("Introduce tu Groq API Key:", type="password")
api_key = api_key_input or os.getenv("GROQ_API_KEY")

if not api_key:
    st.warning("⚠️ Debes introducir tu Groq API Key para usar la aplicación.")
    st.stop()

# --- CARGA DE EXCEL ---
EXCEL_PATH = "Empleados.arff.csv.xlsx"

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return None
    return pd.read_excel(path, engine="openpyxl")

df = load_data(EXCEL_PATH)

if df is None:
    st.error(f"No encuentro el archivo '{EXCEL_PATH}' en el repositorio.")
    st.stop()

with st.expander("👀 Ver muestra del Excel"):
    st.dataframe(df.head(10), use_container_width=True)

st.divider()

# --- CONSULTA ---
question = st.text_input("Haz una pregunta sobre el Excel:", placeholder="Ej: ¿Cuál es el promedio de salario?")

if st.button("Consultar 🤖"):
    if not question:
        st.warning("Escribe una pregunta.")
        st.stop()

    with st.spinner("La IA está analizando los datos..."):
        try:
            llm = ChatGroq(
                groq_api_key=api_key,
                model_name="llama-3.3-70b-versatile",
                temperature=0, # Bajamos a 0 para mayor precisión técnica
            )

            # Cambiamos a OPENAI_FUNCTIONS y añadimos límites de seguridad
            agent = create_pandas_dataframe_agent(
                llm=llm,
                df=df,
                verbose=False,
                allow_dangerous_code=True,
                agent_type=AgentType.OPENAI_FUNCTIONS, # Más rápido y eficiente
                max_iterations=5,  # Evita que el agente se quede en bucle
                handle_parsing_errors=True # Maneja errores de formato en la respuesta
            )

            result = agent.invoke({"input": question})
            
            # Extraer respuesta limpia
            output = result.get("output") if isinstance(result, dict) else str(result)

            st.success("Resultado:")
            st.markdown(output)

        except Exception as e:
            st.error(f"Error al procesar la consulta: {e}")


