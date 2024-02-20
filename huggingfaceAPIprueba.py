from langchain.prompts import PromptTemplate
import streamlit as st
import os
from dotenv import load_dotenv
import requests

load_dotenv()


# Configurar token de API y cabeceras para autenticación
headers = {"Authorization": f"Bearer {os.getenv('huggingface_token')}"}
print(headers)
st.title('🌼 Poem GPT Creator')
prompt = st.text_input('Plug in your prompt here')

# Definir el endpoint de la API para el modelo deseado
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-xl"

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

if prompt:
    # Generar el título del poema
    title_payload = {
        "inputs": f"write me a poem title about {prompt}",
        "parameters": {"temperature": 1.0, "max_length": 100},
        "options": {"use_cache": False},
    }
    title_result = query(title_payload)
    #title_text = title_result[0]['generated_text'] if 'generated_text' in title_result[0] else "Error in generation"
    st.write("Título:", title_result)
    
    # Generar el poema basado en el título
    script_payload = {
        "inputs": f"write me a poem based on this title: {title_result}",
        "parameters": {"temperature": 1.0, "max_length": 100},
        "options": {"use_cache": False},
    }
    script_result = query(script_payload)
    #script_text = script_result[0]['generated_text'] if 'generated_text' in script_result[0] else "Error in generation"
    st.write("Poema:", script_result)