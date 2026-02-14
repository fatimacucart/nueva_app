import os
import streamlit as st
import pandas as pd

from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent

st.set_page_config(page_title="Excel Q&A 🤖", page_icon="📊")
st.title("Habla con tu Excel gracias a la IA")


# -------------------------------
# 🔑 API KEY (obligatoria)
# -------------------------------
st.sidebar.header("🔐 Groq API Key")
api_key = st.sidebar.text_input(
    "Introduce tu Groq API Key:",
    type="password",
    placeholder="gsk-xxxxxxxxxxxxxxxx"
)

if not api_key:
    st.warning("⚠️ Debes introducir tu Groq API Key para usar la aplicación.")
    st.stop()

# Groq suele leer la key desde la variable de entorno
os.environ["GROQ_API_KEY"] = api_key

# -------------------------------
# 📄 Cargar Excel fijo (del repo)
# -------------------------------
EXCEL_PATH = "Empleados.arff.csv.xlsx"  # ⬅️ el nombre exacto del archivo en tu repo/carpeta

if not os.path.exists(EXCEL_PATH):
    st.error(
        f"No encuentro el archivo '{EXCEL_PATH}'. "
        "Asegúrate de que está en la misma carpeta que app.py (o cambia EXCEL_PATH)."
    )
    st.stop()

try:
    df = pd.read_excel(EXCEL_PATH)
except Exception as e:
    st.error(f"No se pudo leer el Excel: {e}")
    st.stop()

# -------------------------------
# 🤖 LLM
# -------------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    max_retries=2,
)

# Agent que traduce preguntas a operaciones sobre pandas
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=False,
    allow_dangerous_code=True,
)

# -------------------------------
# 👀 Vista previa
# -------------------------------
with st.expander("👀 Ver muestra del Excel"):
    st.write(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
    st.dataframe(df.head(20), use_container_width=True)

st.divider()

# -------------------------------
# 💬 Preguntas del usuario
# -------------------------------
st.subheader("Haz una pregunta sobre el Excel")
question = st.text_input(
    "Ejemplos: “¿Cuántos empleados hay?”, “media de salario por departamento”, “top 5 ciudades”…"
)

if st.button("Consultar 🤖"):
    if not question.strip():
        st.warning("Escribe una pregunta primero.")
        st.stop()

    with st.spinner("Analizando..."):
        try:
            result = agent.invoke(question)
            if isinstance(result, dict) and "output" in result:
                st.markdown(result["output"])
            else:
                st.write(result)
        except Exception as e:
            st.error(f"Error consultando el Excel: {e}")




