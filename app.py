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

if not os.path.exists(EXCEL_PATH):
    st.error(f"No encuentro el archivo '{EXCEL_PATH}' en el repositorio.")
    st.stop()

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl")

try:
    df = load_data(EXCEL_PATH)
except Exception as e:
    st.error(f"Error al leer el Excel: {e}")
    st.stop()

with st.expander("👀 Ver muestra del Excel"):
    st.dataframe(df.head(10), use_container_width=True)

st.divider()

# --- CONSULTA ---
question = st.text_input("Haz una pregunta sobre el Excel:")

if st.button("Consultar 🤖"):
    if not question:
        st.warning("Escribe una pregunta.")
        st.stop()

    with st.spinner("Analizando..."):
        try:
            llm = ChatGroq(
                groq_api_key=api_key,
                model_name="llama-3.3-70b-versatile",
                temperature=0.2,
            )

            agent = create_pandas_dataframe_agent(
                llm=llm,
                df=df,
                verbose=False,
                allow_dangerous_code=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            )

            result = agent.invoke({"input": question})

            # LangChain a veces devuelve dict con "output" y a veces texto/objeto
            output = result.get("output") if isinstance(result, dict) else str(result)

            st.success("Resultado:")
            st.markdown(output)

        except Exception as e:
            st.error(f"Error consultando el Excel: {e}")


