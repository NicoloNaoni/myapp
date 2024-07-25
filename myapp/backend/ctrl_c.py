
import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from fastapi.responses import JSONResponse
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_pinecone import PineconeVectorStore
from pydantic import BaseModel
from typing import List, Dict
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
import cohere
from langchain.chains import ConversationalRetrievalChain

load_dotenv("/myapp/backend/.env")

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
        return {"status": "file uploaded and vectorized"}
    except Exception as e:
        print(f"File Upload Error: {str(e)}")  # Log error
        return JSONResponse(status_code=500, content={"error": str(e)})


class QARequest(BaseModel):
    query: str
    history: List[Dict[str, str]]


@app.post("/qa-request")
async def qa_request(qa_request: QARequest):
    try:
        query = qa_request.query
        history = qa_request.history

        chat_history = "\n".join([f"user: {entry.get('user', '')}\nbot: {entry.get('bot', '')}" for entry in history])

        template = '''
            You are a helpful assistant. Answer the following question considering the history of the conversation.

            Chat history: {chat_history}

            User question: {user_question}
            '''
        formatted_prompt = template.format(chat_history=chat_history, user_question=query)
        co = cohere.Client(cohere_api_key)
        response = co.generate(
            model=os.environ['MODEL'],
            prompt=formatted_prompt,
            max_tokens=int(os.environ['MAX_TOKENS'])
        )

        answer = response.generations[0].text.strip()

        new_entry = {"user": query, "bot": answer}
        history.append(new_entry)

        return {"result": answer}
    except Exception as e:
        print(f"QA Request Error: {str(e)}")  # Log error
        return JSONResponse(status_code=500, content={"error": str(e)})


class QADocumentRequest(BaseModel):
    query: str
    history: List[Dict[str, str]]



@app.post("/qa-documents")
async def qa_documents(qa_document_request: QADocumentRequest):
    try:
        query = qa_document_request.query
        history = qa_document_request.history

        llm = ChatCohere(model="command-r-plus", temperature=0.75)
        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 10})
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        crc = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
        answer = crc.invoke({"question": query})  # Pass 'question' key instead of 'query'

        return {"result": answer}
    except Exception as e:
        print(f"QA Document Request Error: {str(e)}")  # Log error
        return JSONResponse(status_code=500, content={"error": str(e)})
