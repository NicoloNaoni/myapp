from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader


load_dotenv("/Users/niko/Desktop/Links Internship/LinksProject/myapp/backend/.env")
pinecone_api_key = os.environ['PINECONE_API_KEY']
index_name = os.environ['INDEX_NAME']
chunk_size = int(os.environ['CHUNK_SIZE'])
chunk_overlap = int(os.environ['CHUNK_OVERLAP'])
cohere_api_key = os.environ['COHERE_API_KEY']

docs = "/Users/niko/Desktop/Links Internship/LinksProject/myapp/docs"

pinecone = Pinecone(api_key=pinecone_api_key)
index = pinecone.Index(index_name)


existing_indexes = pinecone.list_indexes().names()
if index_name in existing_indexes:
    print(f"Deleting {index_name} index")
    pinecone.delete_index(index_name)
    print(f"Index '{index_name}' has been deleted.")
else:
    print(f"Index '{index_name}' does not exist.")

if index_name not in pinecone.list_indexes():
    print(f'Reloading index {index_name}...')
    pinecone.create_index(
        index_name,
        dimension=int(os.environ['DIMENSIONS']), # use toml
        metric=os.environ['METRIC'],
        spec=ServerlessSpec(
            cloud=os.environ['CLOUD'],
            region=os.environ['REGION']
        )
    )


def list_files_in_directory(docs):
    files = os.listdir(docs)

    for file_name in files:
        file_path = os.path.join(docs, file_name)
        if os.path.isfile(file_path):
            vector_store = vectorize(file_path)


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


list_files_in_directory(docs)
print(f"The index has been updated and reloaded")

