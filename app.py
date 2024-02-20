
from langchain.prompts import PromptTemplate
from langchain import HuggingFaceHub
from langchain.chains import LLMChain, SequentialChain
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv('huggingface_token')


st.title('🦜️🔗YouTube GPT Creator')
prompt = st.text_input('Plug in your prompt here')

# Prompt templates
title_templete = PromptTemplate(
    input_variables=['topic'],
    template = 'write me a youtube video title about {topic}'
)

script_templete = PromptTemplate(
    input_variables=['title'],
    template = 'write me a youtube video script base on this title: {title}'
)

# Model
@st.cache_resource
def load_llm():
    return HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 1.0, "max_length": 512})

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

#import streamlit as st
#from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
#from langchain.prompts import PromptTemplate
#from langchain.chains import LLMChain
#
#
#st.title('🦜️🔗YouTube GPT Creator')
#
#prompt = st.text_input('Plug in your prompt here')
#
#title_templete = PromptTemplate(
#    input_variables=['topic'],
#    template = 'write me a youtube video title about {topic}'
#)
#
#
## Utilizaremos st.cache_resource para cargar el modelo y el tokenizador
#@st.cache_resource
#def load_model():
#    return AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
#
#@st.cache_resource
#def load_tokenizer():
#    return AutoTokenizer.from_pretrained("google/flan-t5-xl")
#
#tokenizer = load_tokenizer()
#model = load_model()
#
#def llm(prompt):
#    inputs = tokenizer.encode(prompt, return_tensors='pt')
#    outputs = model.generate(inputs, max_length=100, temperature=1.0)
#    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#    return response
#
#title_chain = LLMChain(llm=llm, prompt = title_templete, verbose=True)
#
#
#if prompt:
#    response = title_chain.run(topic=prompt)
#    st.write(response)
