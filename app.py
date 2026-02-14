import os
import time
import streamlit as st
import pandas as pd

from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentType

# -----------------------------
# CONFIG STREAMLIT
# -----------------------------
st.set_page_config(page_title="Excel AI 🤖", page_icon="📊")
st.title("💬📂 Habla con tu Excel gracias a la IA")

# -----------------------------
# SIDEBAR: API KEY
# -----------------------------
st.sidebar.header("🔐 Configuración")
api_key_input = st.sidebar.text_input("Introduce tu Groq API Key:", type="password")
api_key = api_key_input or os.getenv("GROQ_API_KEY")

if not api_key:
    st.warning("⚠️ Debes introducir tu Groq API Key para usar la aplicación.")
    st.stop()

# -----------------------------
# CARGA DE EXCEL
# -----------------------------
EXCEL_PATH = "Empleados.arff.csv.xlsx"

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_excel(path, engine="openpyxl")

try:
    df = load_data(EXCEL_PATH)
except FileNotFoundError:
    st.error(f"No encuentro el archivo '{EXCEL_PATH}' en el repositorio.")
    st.stop()
except Exception as e:
    st.error(f"Error leyendo el Excel: {e}")
    st.stop()

with st.expander("👀 Ver muestra del Excel"):
    st.dataframe(df.head(10), use_container_width=True)

st.divider()

# -----------------------------
# AGENTE (cacheado para no recrearlo en cada click)
# -----------------------------
@st.cache_resource(show_spinner=False)
def build_agent(_df: pd.DataFrame, _api_key: str):
    llm = ChatGroq(
        groq_api_key=_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.0,
    )

    # En Groq suele ir más estable/rápido este tipo de agente que OPENAI_FUNCTIONS
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=_df,
        verbose=False,
        allow_dangerous_code=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=3,            # menos vueltas = más rápido
        handle_parsing_errors=True,  # evita petadas por formatos raros
    )
    return agent

# -----------------------------
# CONSULTA
# -----------------------------
question = st.text_input(
    "Haz una pregunta sobre el Excel:",
    placeholder="Ej: ¿Cuál es el promedio de salario? ¿Cuántos empleados hay por departamento?"
)

col1, col2 = st.columns([1, 1])
with col1:
    run_btn = st.button("Consultar 🤖", use_container_width=True)
with col2:
    show_timing = st.checkbox("Mostrar tiempos", value=False)

if run_btn:
    if not question.strip():
        st.warning("Escribe una pregunta.")
        st.stop()

    with st.spinner("La IA está analizando los datos..."):
        t0 = time.time()
        try:
            agent = build_agent(df, api_key)
            t_agent = time.time()

            result = agent.invoke({"input": question})
            t_done = time.time()

            output = result.get("output") if isinstance(result, dict) else str(result)

            st.success("Resultado:")
            st.markdown(output)

            if show_timing:
                st.info(
                    f"⏱️ Tiempos\n"
                    f"- Crear/recuperar agente: {t_agent - t0:.2f}s\n"
                    f"- Ejecutar consulta: {t_done - t_agent:.2f}s\n"
                    f"- Total: {t_done - t0:.2f}s"
                )

        except Exception as e:
            st.error(f"Error al procesar la consulta: {e}")



