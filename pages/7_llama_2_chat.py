# import openai
import streamlit as st
from utils.llm import get_pipeline
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain


prompt_area="""
{final_input}
"""
prompt_msg ="""
<s>[INST] <<SYS>>
You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as Assistant."
<</SYS>>
"""
llm_pipeline = get_pipeline()
llm = HuggingFacePipeline(pipeline=llm_pipeline)
prompt = PromptTemplate(template=prompt_area, input_variables=["final_input"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

def format_message(messages):
    query = prompt_msg + f"{messages[0]['content']} [/INST]"
    for i,message in enumerate(messages[1:]):
        if i%2 ==0:
            query +=  f"{message['content']} </s>"
        else:
             query +=  f"[INST] {message['content']} [/INST] "
           
    return query



st.title("ChatGPT-like ChatBot")

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            messages = st.session_state.get("messages")
            full_response = llm_chain.run(final_input= format_message(messages)) 
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})