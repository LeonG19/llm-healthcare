import pandas as pd
import re
from collections import UserList
import requests
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface.llms import HuggingFacePipeline
import faiss
from transformers import AutoTokenizer,BitsAndBytesConfig, pipeline
from transformers import AutoModelForCausalLM
import torch
import os
from torch import cuda
import transformers
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from uuid import uuid4
from langchain_core.documents import Document
import json
def process_and_upsert_json_files(directory):
    data = []

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                content = json.load(file)
                text_chunk = content['chunk']

                metadata = {
                    "id": content.get("id", ""),
                    "title": content.get("title", ""),
                    "url": content.get("url", ""),
                    "source": content.get("source", "")
                }

                document = Document(page_content=text_chunk, metadata=metadata)
                data.append(document)

    
    uuids = [str(uuid4()) for _ in range(len(data))]
    return data, uuids
class RAG_Healthcare:
    def __init__(self):
        data, uuids = process_and_upsert_json_files("vector_db_data")

        #Load embedding function

        key = '' #You have to provide you own hugging face api key here. Make sure you have access to llama-3 models (you have to request it).
        embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

        device = torch.device(f"cuda:{1}")

        embed_model = HuggingFaceEmbeddings(
            model_name=embed_model_id,
            model_kwargs={'device': device},
            encode_kwargs={'device': device, 'batch_size': 32}
        )

        #Create vector database
        vectorstore_Faiss= FAISS(embedding_function=embed_model, index=faiss.IndexFlatL2(len(embed_model.embed_query("hello world"))), docstore= InMemoryDocstore(),  index_to_docstore_id={} )
        vectorstore_Faiss.add_documents(data, ids=uuids)

        #Load LLM with huggingface, utilize quantization for reduced model size, faster inference but lower accuracy


        my_secret_key = key
        print("using", device)
        model_id = "meta-llama/Llama-3.1-8B-Instruct"
        llm = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, use_auth_token = key)
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token = key, max_length=200)
        llm_pipeline = pipeline(
            "text-generation",
            model=llm,
            tokenizer=tokenizer,
            max_new_tokens=200,
            do_sample=False,  # Disable random sampling (greedy decoding)
            eos_token_id=tokenizer.eos_token_id,  # Stop when the end-of-sequence token is generated
            pad_token_id=tokenizer.eos_token_id   # Avoid the warning message
        )
        self.llm_pipeline = HuggingFacePipeline(pipeline = llm_pipeline)

        #Create RAG chain

        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        combine_docs_chain = create_stuff_documents_chain(self.llm_pipeline, retrieval_qa_chat_prompt)
        self.rag_chain = create_retrieval_chain(vectorstore_Faiss.as_retriever(), combine_docs_chain)
    
    def ask_question_RAG(self, question):
        answer = self.rag_chain.invoke({"input": question})['result']
        return answer
    def ask_question_LLM(self, question):
        answer = self.llm_pipeline.invoke({"input": question})