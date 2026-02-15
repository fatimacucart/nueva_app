import os
import streamlit as st
import pandas as pd

from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits.pandas.base import (
    create_pandas_dataframe_agent,
)

# -----------------------------
# CONFIG STREAMLIT
# -----------------------------
st.set_page_config(page_title="Excel Q&A 🤖", page_icon="📊")
st.title("💬📂 Habla con tu Excel gracias a la IA")

# -----------------------------
# API KEY
# -----------------------------
st.sidebar.header("🔐 Groq API Key")
api_key = st.sidebar.text_input(
    "Introduce tu Groq API Key:",
    type="password",
)

if not api_key:
    st.warning("⚠️ Debes introducir tu Groq API Key.")
    st.stop()

# -----------------------------
# CARGA EXCEL
# -----------------------------
EXCEL_PATH = "Empleados.arff.csv.xlsx"

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl")

if not os.path.exists(EXCEL_PATH):
    st.error(f"No encuentro el archivo '{EXCEL_PATH}'.")
    st.stop()

try:
    df = load_data(EXCEL_PATH)
except Exception as e:
    st.error(f"No se pudo leer el Excel: {e}")
    st.stop()

with st.expander("👀 Ver muestra del Excel"):
    st.write(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
    st.dataframe(df.head(20), use_container_width=True)

st.divider()

# -----------------------------
# LLM + AGENTE
# -----------------------------
@st.cache_resource
def build_agent(_df: pd.DataFrame, _api_key: str):
    llm = ChatGroq(
        api_key=_api_key,
        model="llama-3.3-70b-versatile",
        temperature=0.2,
    )

    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=_df,
        agent_type="zero-shot-react-description",  # ✅ ESTE ES EL CORRECTO
        verbose=False,
        allow_dangerous_code=True,
        max_iterations=10,
        handle_parsing_errors=True,
    )
    return agent

agent = build_agent(df, api_key)

# -----------------------------
# PREGUNTA
# -----------------------------
question = st.text_input("Haz una pregunta sobre el Excel:")

if st.button("Consultar 🤖"):
    if not question.strip():
        st.warning("Escribe una pregunta.")
        st.stop()

    with st.spinner("Analizando..."):
        try:
            result = agent.invoke({"input": question})
            output = result.get("output") if isinstance(result, dict) else str(result)

            st.success("Resultado:")
            st.markdown(output)

        except Exception as e:
            st.error(f"Error consultando el Excel: {e}")




