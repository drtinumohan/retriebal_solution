from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import streamlit as st
import pandas as pd


def get_qa_model(model_name):  # ="deepset/roberta-base-squad2-distilled"):
    nlp = pipeline("question-answering", model=model_name, tokenizer=model_name)
    return nlp


def get_anwser(nlp, query, relevent_docs):
    QA_input = [
        {
            "question": query,  #'Why is model conversion important?',
            "context": doc,  #'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
        }
        for doc in relevent_docs
    ]
    res = nlp(QA_input)
    return res


def format_qa_output(relevent_docs, document_relevance_scores, answer):
    __ = {
        key.update(
            {
                "paragraph": relevent_docs.pop(0),
                "re_score": document_relevance_scores.pop(0),
                "qa_score": round(key["score"], 3),
            }
        )
        for key in answer
    }
    answer_df = pd.DataFrame(answer)
    return answer_df.sort_values(["score"], ascending=False)
