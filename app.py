#
## Bring in deps
#import os
#from dotenv import load_dotenv
#import streamlit as st
#from langchain.llms import huggingface_hub
#from langchain.prompts import PromptTemplate
#from langchain.chains import LLMChain, SequentialChain 
#from langchain.memory import ConversationBufferMemory
#from langchain.utilities import WikipediaAPIWrapper 
#load_dotenv()
#
#
#os.environ['HUGGINGFACE_API_TOKEN'] = os.getenv('huggingface_token')
#
#
#
## App framework
#st.title('Youtube GPT Creator')
#prompt = st.text_input('Plug in your prompt here')
#
## Llms
#llm = huggingface_hub(repo_id = "google/flan-t5-xl",model_kwargs={"temperature":0})
#
##Show stuff to the screeen of theres a prompt
#if prompt:
#    response = llm(prompt)
#    st.write(response)
#

import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Establecer el título de la app
st.title('YouTube GPT Creator')

# Entrada de texto para el prompt
prompt = st.text_input('Plug in your prompt here')

# Cargar el tokenizador y el modelo desde Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# Función para generar texto a partir del prompt
def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=100, temperature=0)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Mostrar la respuesta en la pantalla si hay un prompt
if prompt:
    response = generate_response(prompt)
    st.write(response)