from sentence_transformers import SentenceTransformer, util
import numpy as np
import faiss
import uuid


def get_model(model_name):  # = 'sentence-transformers/all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    return model


def get_embedding(model, arr):
    return model.encode(arr)


def compute_score(doc_emb, query_emb, top_k=3):
    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
    return [docs[i] for i in np.argsort(scores)[-top_k:]]


# def get_top_k_result(doc_emb, query_emb, docs, top_k=3):
#     # Compute dot score between query and all document embeddings
#     scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
#     return [(docs[i], scores[i]) for i in np.argsort(scores)[-5:]]


def get_fasis_model(doc_emb):
    index = faiss.IndexFlatIP(len(doc_emb[0]))
    vector = np.array(doc_emb, dtype=np.float32)
    index.add(vector)
    return index

def get_top_k_result(index_model, query_emb, docs, top_k=3):
    docs_score, doc_index = index_model.search(np.array([query_emb], dtype=np.float32), k=top_k)
    docs_score = docs_score.tolist()[0]
    print(doc_index)
    return [(docs[val], docs_score[i]) for i,val in enumerate(doc_index.tolist()[0])]

