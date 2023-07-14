

from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import streamlit as st
import torch
model = "tiiuae/falcon-7b-instruct" #tiiuae/falcon-40b-instruct
@st.cache_resource
def get_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(model)
    llm_pipeline = pipeline(
        "text-generation", #task
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=512,
        do_sample=False,
        top_k=1,
        temperature=0.1             ,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )
    # llm_pipeline=AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
    return llm_pipeline