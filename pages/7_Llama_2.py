# python 3.8 (3.8.16) or it doesn't work
# pip install streamlit streamlit-chat langchain python-dotenv
import streamlit as st
from streamlit_chat import message
import os
from utils.llm import get_pipeline
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
prompt_area="""
{final_input}
"""
styl = f"""
<style>
    .stTextInput {{
      position: fixed;
      bottom: 3rem;
    }}
</style>
"""
st.markdown(styl, unsafe_allow_html=True)


def format_message(messages, responses, user_input):
    if len(messages)==0:
        return prompt_msg + f"{user_input} [/INST]"
    query = prompt_msg + f"{messages[0]} [/INST] {responses[0]} </s>"
    for i,history in enumerate(messages[1:]):
        query += f"<s>[INST] {history} [/INST] {responses[i]} </s>"
    query += f"<s>[INST] {user_input} [/INST]"
    return query




# chat = ChatOpenAI(temperature=0)

# initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = [    
    ]
if "responses" not in st.session_state:
    st.session_state.responses = [    
    ]

user_text_input = st.empty()
user_input = user_text_input.text_input("Your message: ", key="user_input")#get_text()
llm_pipeline = get_pipeline()
llm = HuggingFacePipeline(pipeline=llm_pipeline)
prompt_msg ="""
<s>[INST] <<SYS>>
You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as Assistant."
<</SYS>>
"""
prompt = PromptTemplate(template=prompt_area, input_variables=["final_input"])
llm_chain = LLMChain(prompt=prompt, llm=llm)
# handle user input
if user_input:
    messages = st.session_state.get('messages')
    responses = st.session_state.get('responses')
    final_input = format_message(messages,responses, user_input)
    print("\n\n",final_input)
    st.session_state.messages.append(user_input)#HumanMessage(content=user_input))
    
    with st.spinner("Thinking..."):
        response = llm_chain.run(final_input= final_input)
    st.session_state.responses.append(response)#AIMessage(content= response))

# display message history
messages = st.session_state.get('messages')
responses = st.session_state.get('responses')
for i, msg in enumerate(messages):
    message(msg, is_user=True, key=str(i) + '_user')
    message(responses[i], is_user=False, key=str(i) + '_ai')

