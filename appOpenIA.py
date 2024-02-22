
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.llms import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper


os.environ["OPENAI_API_KEY"] = os.getenv('openaikey')

st.title('🌼 Poem GPT Creator')
prompt = st.text_input('Plug in your prompt here')

title_template = PromptTemplate(
    input_variables = ['topic'],
    template = 'write me a poem title about {topic}'
)

script_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'],
    template = 'write me a poem of two verses base on this title: {title} while leveraging this wikipedia reserch:{wikipedia_research}'
)


title_memory = ConversationBufferMemory(input_key = 'topic', momery_key='chat_history')
script_memory = ConversationBufferMemory(input_key = 'title', momery_key='chat_history')

llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True,
                        output_key = 'title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, 
                        output_key = 'script',memory=script_memory)
#sequentialchain = SequentialChain(chains=[title_chain, script_chain], input_variables = ['topic'],
#                                       output_variables=['title','script'], verbose=True)

wiki = WikipediaAPIWrapper()

if prompt:
    title=title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script=script_chain.run(title=title, wikipedia_research=wiki_research)
    st.write(title)
    st.write(script)


    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('Scirpt History'):
        st.info(script_memory.buffer)

    with st.expander('wikipedia_research'):
        st.info(wiki_research)