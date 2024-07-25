import streamlit as st
import requests
from langchain_core.messages import HumanMessage, AIMessage
import os

st.set_page_config(page_title="My Chatbot", page_icon="🤖")
st.title("My Chatbot")

# Function to handle file upload
def upload_file(file):
    data = file.read()
    save_path = "/Users/niko/Desktop/Links Internship/LinksProject/myapp/docs"
    file_path = os.path.join(save_path, file.name)

    if file.name not in save_path:
        with open(file_path, "wb") as f:
            f.write(data)

    response = requests.post(
        "http://localhost:8000/upload-file", files={"file": data}
    )
    return response.json()

# File upload section
uploaded_file = st.file_uploader("Upload a document to ask specific questions", type=["txt"])
if uploaded_file is not None:
    upload_file(uploaded_file)

def get_response(query, chat_history, endpoint):
    payload = {
        "query": query,
        "history": []
    }

    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            payload["history"].append({"user": msg.content, "bot": ""})
        else:
            payload["history"].append({"user": "", "bot": msg.content})
    response = requests.post(endpoint, json=payload)
    response_data = response.json()
    print(response_data)  # Print full response for debugging
    return response_data.get("result", "Error in response")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Toggle for query type
query_type = st.radio("Choose query type:", ('General', 'Document-Specific'))

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

# Single chat input
user_query = st.chat_input("Your message")
if user_query:
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    endpoint = "http://127.0.0.1:8000/qa-request" if query_type == 'General' else "http://127.0.0.1:8000/qa-documents"

    with st.chat_message("AI"):
        ai_response = get_response(user_query, st.session_state.chat_history, endpoint)
        st.markdown(ai_response)  # Directly display the answer

    # Append the AIMessage to the session state chat history
    st.session_state.chat_history.append(AIMessage(content=ai_response))
