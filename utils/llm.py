

from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, LlamaTokenizer
import streamlit as st
import torch
model = "tiiuae/falcon-7b-instruct" #tiiuae/falcon-40b-instruct
@st.cache_resource
def get_pipeline():
    # tokenizer = AutoTokenizer.from_pretrained(model)
    # llm_pipeline = pipeline(
    #     "text-generation", #task
    #     model=model,
    #     tokenizer=tokenizer,
    #     torch_dtype=torch.bfloat16,
    #     trust_remote_code=True,
    #     device_map="auto",
    #     # max_length=1024,
    #     max_new_tokens=200,
    #     do_sample=True,
    #     top_k=10,
    #     temperature=0.1,
    #     num_return_sequences=1,
    #     eos_token_id=tokenizer.eos_token_id
    # )
    ## llm_pipeline=AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)

    tokenizer = LlamaTokenizer.from_pretrained("/home/ubuntu/working_nlp/retriebal_solution/retriebal_solution/llama/7b-hf")
    llm_pipeline = pipeline(
        "text-generation", #task
        model="/home/ubuntu/working_nlp/retriebal_solution/retriebal_solution/llama/7b-chat-hf",
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        # max_length=1024,
        max_new_tokens=400,
        do_sample=True,
        top_k=10,
        temperature=0.1,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )
    return llm_pipeline