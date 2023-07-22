from utils.llm import get_pipeline
from langchain import HuggingFacePipeline
from utils.embedding import get_top_k_result
from pathlib import Path
import torch
import streamlit as st
from langchain import PromptTemplate, LLMChain


import json
from collections import deque

def extract_first_json(text):
    json_queue = deque()
    start, end = -1,-1
    for idx,char in enumerate(text):
        if char == '{':
            if start == -1:
                start = idx     
            json_queue.append(char)  
        elif char == '}':
            json_queue.pop()
            if len(json_queue)==0:
                end = idx
                return json.loads(text[start:end+1])

llm_pipeline = get_pipeline()
llm = HuggingFacePipeline(pipeline=llm_pipeline)

#You are a helpful AI assistant and provide >>OUTPUT<< AS JSON for >>QUERY<< by learening patterns >>INPUT<< and >>OUTPUT<< 

example = """
You are a helpful AI assistant. From the examples given below, Learn the  >>OUTPUT<< generated from >>INPUT<<
>>INPUT<< all fights from AYT
>>OUTPUT<< {"logicalOperator":"AND","conditions":[{"entity":"flight","property":"from_airport","condition":"=","value":"AYT"}]}
>>INPUT<< Show me all flights arriving or departing at DXB
>>INPUT<< Show me all flights going through DXB
>>OUTPUT<< {"logicalOperator":"OR","conditions":[{"entity":"flight","property":"from_airport","condition":"=","value":"DXB"},{"entity":"flight","property":"to_airport","condition":"=","value":"DXB"}]}
>>INPUT<< Show all fights from AYT to FRA
>>OUTPUT<< {"logicalOperator":"AND","conditions":[{"entity":"flight","property":"from_airport","condition":"=","value":"AYT"},{"entity":"flight","property":"to_airport","condition":"=","value":"FRA"}]}
>>INPUT<< List all 777 aircraft
>>OUTPUT<< {"logicalOperator":"AND","conditions":[{"entity":"flight","property":"aircraft_type","condition":"=","value":"777"}]}
>>INPUT<< Show me all the flights departing in less than 5 hours
>>OUTPUT<< {"logicalOperator":"AND","conditions":[{"entity":"flight","property":"flight_date_time","condition":"<","value":"5 hours"}]}
>>INPUT<< Show me all flights at DXB between 0300 and 0800 today 
>>OUTPUT<<{"logicalOperator":"AND","conditions":[{"entity":"flight","property":"from_airport","condition":"=","value":"DXB"},{"entity":"flight","property":"flight_date_time","condition":"BETWEEN","value":["0300","0800"]}]}
now provide >>OUTPUT<< AS JSON for the >>QUERY<<
"""

prompt_area = """
{example}
>>QUERY<< {question}
>>OUTPUT<< 
"""



query_area = st.text_area(
    "query",
    value='',
    height=200,
    max_chars=15000,
    key=None,
    help=None,
    on_change=None,
    args=None,
    kwargs=None,
    placeholder=None,
    disabled=False,
    label_visibility="visible",
)

if (
    query_area
):

    query_area = {
        "example":example,
        "question":query_area
    }
    prompt = PromptTemplate(template=prompt_area, input_variables=["example","question"])
    print("query generation\n", (prompt.format(**query_area)))
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    output = llm_chain.run(**query_area)
    print(output)
    st.write(extract_first_json(output))







