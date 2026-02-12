import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="Content Generator 🤖", page_icon="🤖")
st.title("Content generator")

# -------------------------------
# 🔑 API KEY Input (User Option)
# -------------------------------
st.sidebar.header("🔐 Groq API Key")

user_api_key = st.sidebar.text_input(
    "Enter your Groq API Key:",
    type="password",
    placeholder="gsk-xxxxxxxxxxxxxxxx"
)

api_key = user_api_key or os.getenv("GROQ_API_KEY")

if not api_key:
    st.sidebar.warning("⚠️ Please enter your Groq API Key to continue.")
    st.stop()

# ✅ más robusto (Cloud/Local): forzar env var
os.environ["GROQ_API_KEY"] = api_key

# -------------------------------
# Connection with the LLM
# -------------------------------
id_model = "llama-3.3-70b-versatile"

llm = ChatGroq(
    model=id_model,
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# -------------------------------
# Generation function
# -------------------------------
def llm_generate(llm, prompt):
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a digital marketing expert specialized in SEO and persuasive copywriting."),
        ("human", "{prompt}"),
    ])

    chain = template | llm | StrOutputParser()
    res = chain.invoke({"prompt": prompt})
    return res

# -------------------------------
# UI Inputs
# -------------------------------
topic = st.text_input("Topic:", placeholder="e.g., nutrition, mental health...")
platform = st.selectbox("Platform:", ['Instagram', 'Facebook', 'LinkedIn', 'Blog', 'E-mail'])
tone = st.selectbox("Message tone:", ['Normal', 'Informative', 'Inspiring', 'Urgent', 'Informal'])
length = st.selectbox("Text length:", ['Short', 'Medium', 'Long'])
audience = st.selectbox("Target audience:", ['All', 'Young adults', 'Families', 'Seniors', 'Teenagers'])
cta = st.checkbox("Include CTA")
hashtags = st.checkbox("Return Hashtags")
keywords = st.text_area("Keywords (SEO):", placeholder="Example: wellness, preventive healthcare...")

# -------------------------------
# Button Action
# -------------------------------
if st.button("Content generator"):
    prompt = f"""
Write an SEO-optimized text on the topic '{topic}'.
Return only the final text in your response and don't put it inside quotes.
- Platform where it will be published: {platform}.
- Tone: {tone}.
- Target audience: {audience}.
- Length: {length}.
- {"Include a clear Call to Action." if cta else "Do not include a Call to Action."}
- {"Include relevant hashtags at the end of the text." if hashtags else "Do not include hashtags."}
{"- Keywords to include (for SEO): " + keywords if keywords else ""}
"""
    try:
        res = llm_generate(llm, prompt)
        st.markdown(res)
    except Exception as e:
        st.error(f"Error: {e}")

