
from langchain.prompts import PromptTemplate
from langchain import HuggingFaceHub
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv('huggingface_token')


st.title('Poem GPT Creator')
prompt = st.text_input('Plug in your prompt here')

# Prompt templates
title_templete = PromptTemplate(
    input_variables=['topic'],
    template = 'write me a poem title about {topic}'
)

script_templete = PromptTemplate(
    input_variables=['title'],
    template = 'write me a poem base on this title: {title}'
)

# Memory
#memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')

# Model
@st.cache_resource
def load_llm():
    return HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 1.0, "max_length": 100})

# LLms
llm = load_llm()
title_chain = LLMChain(llm=llm, prompt = title_templete, verbose=True, output_key='title')
script_chain = LLMChain(llm=llm, prompt = script_templete, verbose=True, output_key='script')
 #sequential_chain = SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'], 
 #                                  output_variables=['script'], verbose=True)


if prompt:
    title_result = title_chain.run(prompt)
    st.write("Título:", title_result)  # Directamente escribe el resultado sin usar ['output']

    # Ejecuta la cadena para el guion usando el mismo prompt del usuario
    script_result = script_chain.run(title_result)
    st.write("Guion:", script_result) 


"""

*********************************** UTILIZAMOS UN MODEL MÁS GRANDE CARGANDOLO DIRECTAMENTE EN LOCAL DEBIDO A SU TAMAÑO ****************************************

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import streamlit as st

# Define el nombre del modelo que deseas cargar
model_name = "google/flan-t5-xl"

# Función para cargar el tokenizador y el modelo
@st.cache_resource
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# Cargar el tokenizador y el modelo
tokenizer, model = load_model_and_tokenizer(model_name)

# Definir la función para generar texto con el modelo
def generate_text(input_text, max_length=100, temperature=1.0):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=max_length, temperature=temperature)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Usar Streamlit para la interacción con el usuario
st.title('🌼 Poem GPT Creator')
prompt = st.text_input('Plug in your prompt here')

if prompt:
    # Generar el título del poema
    title_text = generate_text(f"write me a youtube video title about {prompt}")
    st.write("Título:", title_text)
    
    # Generar el poema basado en el título
    poem_text = generate_text(f"write me a youtube guion based on this title: {title_text}")
    st.write("Poema:", poem_text)

"""






