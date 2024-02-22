
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.prompts import PromptTemplate
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('openaikey')

st.title('🌼 Poem GPT Creator')
prompt = st.text_input('Plug in your prompt here')

title_template = PromptTemplate(
    input_variables = ['topic'],
    template = 'write me a poem title about {topic}'
)

script_template = PromptTemplate(
    input_variables=['title'],
    template = 'write me a poem base on this title: {title}'
)

llm = OpenAI(temperature=0.9)


title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key = 'title')
script_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key = 'script')
sequentialchain = SimpleSequentialChain(chains=[title_chain, script_chain], input_variables = ['topic'],
                                        output_variables=['title','script'], verbose=True)

if prompt:
    response = sequentialchain.run({'topic':prompt})
    st.write(response['title'])
    st.write(response['script'])