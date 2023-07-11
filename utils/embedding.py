from sentence_transformers import SentenceTransformer, util
import numpy as np


def get_model(model_name = 'sentence-transformers/all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    return model

def get_embedding(model, arr):
    return model.encode(arr)

def compute_score(doc_emb, query_emb, top_k=3):
    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
    return [docs[i] for i in np.argsort(scores)[-top_k:]]

def get_top_k_result(doc_emb, query_emb, docs,top_k=3):
    #Compute dot score between query and all document embeddings
    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
    return [(docs[i],scores[i]) for i in np.argsort(scores)[-5:]]