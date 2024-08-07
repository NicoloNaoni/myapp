import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
from fastapi.responses import JSONResponse
from langchain_cohere import ChatCohere
from langchain.chains import RetrievalQA
import cohere
from pydantic import BaseModel
from typing import List, Dict
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader

load_dotenv("/Users/niko/Desktop/Links Internship/LinksProject/myapp/backend/.env")

cohere_api_key = os.environ['COHERE_API_KEY']
pinecone_api_key = os.environ['PINECONE_API_KEY']
index_name = os.environ['INDEX_NAME']
chunk_size = int(os.environ['CHUNK_SIZE'])
chunk_overlap = int(os.environ['CHUNK_OVERLAP'])
save_path = "/Users/niko/Desktop/Links Internship/LinksProject/myapp/docs"

pinecone = Pinecone(api_key=pinecone_api_key)
embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key)
vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)

def vectorize(file_path):
    if str(file_path).lower().endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        document = loader.load()
    else:
        loader = TextLoader(file_path)
        document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_documents(document)
    embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key)
    vector_store = PineconeVectorStore.from_documents(chunks, index_name=index_name, embedding=embeddings)
    return vector_store

app = FastAPI()

processed_files = set()

@app.post("/upload-file")
async def upload_file(file: UploadFile):
    file_path = os.path.join(save_path, file.filename)

    if os.path.exists(file_path):
        response = {"status": "File already exists"}
    else:
        content = await file.read()

        # Save the file
        with open(file_path, "wb") as f:
            f.write(content)

        # Vectorize the file
        vectorize(file_path)

        response = {"status": "File uploaded and vectorized"}
        return JSONResponse(response)

class QARequest(BaseModel):
    query: str
    history: List[Dict[str, str]]

@app.post("/qa-request")
async def qa_request(qa_request: QARequest):
    query = qa_request.query
    history = qa_request.history

    formatted_history = "\n".join([f"User: {entry['user']}\nBot: {entry['bot']}" for entry in history])
    prompt = f"{formatted_history}\nUser: {query}\nBot:"

    co = cohere.Client(cohere_api_key)
    response = co.generate(
        model=os.environ['MODEL'],
        prompt=prompt,
        max_tokens=int(os.environ['MAX_TOKENS']),
        return_likelihoods='GENERATION'
    )


    best_generation = None
    best_logprob = float('-inf')

    for generation in response.generations:
        generation_text = generation.text.strip()
        generation_logprob = generation.likelihood

        if generation_logprob > best_logprob:
            best_generation = generation_text
            best_logprob = generation_logprob

    answer = best_generation

    return {"result": answer}

class QADocumentRequest(BaseModel):
    query: str
    history: List[Dict[str, str]]

@app.post("/qa-documents")
async def qa_documents(qa_document_request: QADocumentRequest):
    query = qa_document_request.query
    history = qa_document_request.history

    formatted_history = "\n".join([f"User: {entry['user']}\nBot: {entry['bot']}" for entry in history])
    prompt = (
        f"""Only respond if you are provided with context. If no context is given, simply state that you don’t know. 
        In your response, do not mention the use of context. For each new query, answer only the most recent one while 
        considering previous queries. Provide precise, fluent, and elaborate answers.
        {formatted_history}\nUser: {query}\nBot:"""
    )

    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    llm = ChatCohere(model=os.environ['MODEL'], temperature=float(os.environ['TEMPERATURE']))
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': int(os.environ['SEARCH_KWARGS'])})
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=retriever, combine_docs_chain=combine_docs_chain
    )
    answer_dict = retrieval_chain.invoke({"input": prompt})
    value = answer_dict["answer"]

    sources = [doc.metadata.get('source') for doc in answer_dict.get('context', [])]

    return {"result": value, "sources": sources}