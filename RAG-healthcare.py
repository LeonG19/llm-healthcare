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
#Load parser


#Create chunks and prepare list of documents to be used
def split_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    chunks = content.split('|')
    return [chunk.strip() for chunk in chunks if chunk.strip()]

#Prepare chunks as documents so they can be upserted to the db
data = []
raw_chunks = split_text("Geriatric_ECHO,_01_08_2020___Cancer_Screening_in_Older_Adults_20250604_1653_chunks.txt")
for chunk in raw_chunks:
   data.append(Document(page_content=chunk, metadata = {'source':"Geriatric_ECHO,_01_08_2020___Cancer_Screening_in_Older_Adults_20250604_1653_chunks"}))
uuids = [str(uuid4()) for _ in range(len(data))]

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
llm_pipeline = HuggingFacePipeline(pipeline = llm_pipeline)

#Create RAG chain

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
combine_docs_chain = create_stuff_documents_chain(llm_pipeline, retrieval_qa_chat_prompt)
rag_chain = create_retrieval_chain(vectorstore_Faiss.as_retriever(), combine_docs_chain)

print(rag_chain.invoke({"input": "What should you do if a person with dementia wanders away?"})['result'])
