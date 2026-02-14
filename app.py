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
api_key_input = st.sidebar.text_input(
    "Introduce tu Groq API Key:",
    type="password",
    placeholder="gsk-xxxxxxxxxxxxxxxx"
)

# Prioridad: Input manual o variable de entorno
api_key = api_key_input or os.getenv("GROQ_API_KEY")

if not api_key:
    st.warning("⚠️ Debes introducir tu Groq API Key en la barra lateral para usar la aplicación.")
    st.stop()

# -------------------------------
# 📄 Cargar Excel fijo
# -------------------------------
EXCEL_PATH = "Empleados.arff.csv.xlsx"

if not os.path.exists(EXCEL_PATH):
    st.error(f"No encuentro el archivo '{EXCEL_PATH}'.")
    st.stop()

try:
    df = pd.read_excel(EXCEL_PATH)
except Exception as e:
    st.error(f"No se pudo leer el Excel: {e}")
    st.stop()

# -------------------------------
# 👀 Vista previa
# -------------------------------
with st.expander("👀 Ver muestra del Excel"):
    st.write(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
    st.dataframe(df.head(10), use_container_width=True)

st.divider()

# -------------------------------
# 💬 Preguntas del usuario
# -------------------------------
st.subheader("Haz una pregunta sobre el Excel")
question = st.text_input("Ejemplo: ¿Cuál es el salario medio?")

if st.button("Consultar 🤖"):
    if not question.strip():
        st.warning("Escribe una pregunta primero.")
    else:
        with st.spinner("Analizando..."):
            try:
                # 🤖 CREAMOS EL LLM Y EL AGENTE AQUÍ ADENTRO
                # Solo cuando el usuario hace clic, asegurando que la API KEY ya existe.
                llm = ChatGroq(
                    groq_api_key=api_key, # Pasamos la clave directamente
                    model="llama-3.3-70b-versatile",
                    temperature=0.2
                )

                agent = create_pandas_dataframe_agent(
                    llm,
                    df,
                    verbose=False,
                    allow_dangerous_code=True,
                )

                result = agent.invoke(question)
                
                if isinstance(result, dict) and "output" in result:
                    st.success("Resultado:")
                    st.markdown(result["output"])
                else:
                    st.write(result)
            except Exception as e:
                st.error(f"Error consultando el Excel: {e}")

