#!/usr/bin/env python
# coding: utf-8

# In[56]:


from utils.converter import pdf_converter
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd


# In[2]:


array_list = pdf_converter("documents/", min_length=200)


# In[3]:


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

docs = array_list
doc_emb = model.encode(docs)


# In[4]:


doc_emb.shape,len(docs)


# In[5]:


#Load the model


#Encode query and documents


def compute_score(doc_emb, query_emb, top_k=3):
    #Compute dot score between query and all document embeddings
    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
    return [docs[i] for i in np.argsort(scores)[-5:]]
    #Combine docs & scores
#     doc_score_pairs = list(zip(docs, scores))
    #Sort by decreasing score
#     doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
#     return doc_score_pairs
#     for doc, score in doc_score_pairs[0:5]:
#         print(score, doc)
#     return [doc for doc,score in doc_score_pairs[:top_k]]
#     return doc_score_pairs
#Output passages & scores



# In[34]:


from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name ="deepset/roberta-base-squad2-distilled" #
# model_name="deepset/roberta-base-squad2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

# a) Get predictions


# In[67]:


def qa_model(query, relevent_docs):
    QA_input = [{
        'question': query,#'Why is model conversion important?',
        'context': doc#'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
    } for doc in relevent_docs]
    res = nlp(QA_input)
    return res

# b) Load model & tokenizer
# model = AutoModelForQuestionAnswering.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
def get_answer(query):
    query_emb = model.encode(query)
    relevent_docs = compute_score(doc_emb, query_emb,top_k=5)
    answer = qa_model(query, relevent_docs)
    __  = {key.update({"paragraph":relevent_docs.pop(0), "score":round(key["score"], 3)}) for key in answer }
    answer_df = pd.DataFrame(answer)
    return answer_df.sort_values(["score"],ascending= False)


# In[68]:


# for doc in doc_score_pairs[0:5]:
#     print(doc)
temp = get_answer("where is he born")


# In[69]:


temp


# In[70]:


# query = "which district was  he born"
# query_emb = model.encode(query)

# relevent_docs = compute_score(doc_emb, query_emb,top_k=5)
# relevent_docs


# In[ ]:





# In[ ]:




