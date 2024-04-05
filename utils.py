from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document
import pinecone
from langchain_openai import OpenAIEmbeddings
from pypdf import PdfReader
from langchain.vectorstores import Pinecone, Chroma
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import HuggingFaceHub
from langchain_pinecone import PineconeVectorStore

import time


def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def create_docs(user_pdf_list, unique_id):
    docs=[]
    for filename in user_pdf_list:
        
        chunks=get_pdf_text(filename)
        docs.append(Document(
            page_content=chunks,
            metadata={"name": filename.name,"id":filename.file_id,"type=":filename.type,"size":filename.size,"unique_id":unique_id},
        ))

    return docs

def create_embeddings_load_data():
    embeddings = OpenAIEmbeddings()
    return embeddings

def push_to_pinecone(pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings,docs):

    
    docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=pinecone_index_name)

def pull_from_pinecone(pinecone_api_key,pinecone_environment,pinecone_index_name,embeddings):
    
    print("20secs delay...")
    time.sleep(20)

    index_name = pinecone_index_name

    index = Pinecone.from_existing_index(index_name, embeddings)
    return index

def similar_docs(query,k,pinecone_api_key,pinecone_environment,pinecone_index_name,embeddings,unique_id):

    index_name = pinecone_index_name

    index = pull_from_pinecone(pinecone_api_key,pinecone_environment,index_name,embeddings)
    similar_docs = index.similarity_search_with_score(query, int(k),{"unique_id":unique_id})
    #print(similar_docs)
    return similar_docs

def get_summary(current_doc):
    llm = ChatOpenAI(model_name="gpt-4-turbo-preview")  
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run([current_doc])

    return summary