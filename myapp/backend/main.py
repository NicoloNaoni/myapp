import os
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from fastapi import UploadFile
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
from fastapi.responses import JSONResponse
from langchain_cohere import ChatCohere
from langchain.chains import RetrievalQA
import cohere
from pydantic import BaseModel
from typing import List, Dict

load_dotenv("/Users/niko/Desktop/Links Internship/LinksProject/myapp/backend/.env")

cohere_api_key = os.environ['COHERE_API_KEY']
pinecone_api_key = os.environ['PINECONE_API_KEY']
index_name = os.environ['INDEX_NAME']
chunk_size = int(os.environ['CHUNK_SIZE'])
chunk_overlap = int(os.environ['CHUNK_OVERLAP'])

pinecone = Pinecone(api_key=pinecone_api_key)
embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key)
vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)

def vectorize(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = text_splitter.create_documents([text])
        embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key)
        vector_store = PineconeVectorStore.from_documents(chunks, index_name=index_name, embedding=embeddings)
        return vector_store
    except Exception as e:
        raise RuntimeError(f"Error during vectorization: {str(e)}")

app = FastAPI()

@app.post("/upload-file")
async def upload_file(file: UploadFile):
    try:
        content = await file.read()
        text_content = content.decode("utf-8")
        vector_store = vectorize(text_content)
        response = {"status": "File uploaded and vectorized"}
        return response["status"]
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

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
        generation_logprob = generation.likelihood  # log probability of the generation

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

    # Format the history to avoid duplicates
    formatted_history = "\n".join([f"User: {entry['user']}\nBot: {entry['bot']}" for entry in history])
    prompt = f"{formatted_history}\nUser: {query}\nBot:"

    llm = ChatCohere(model=os.environ['MODEL'], temperature=int(os.environ['TEMPERATURE']))
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': int(os.environ['SEARCH_KWARGS'])})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
    answer_dict = chain.invoke({"query": prompt})
    value = answer_dict["result"]

    return {"result": value}

