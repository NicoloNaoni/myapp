import streamlit as st
import requests
from langchain_core.messages import HumanMessage, AIMessage
import os

save_path = "/Users/niko/Desktop/Links Internship/LinksProject/myapp/docs"

st.set_page_config(page_title="My Chatbot", page_icon="🤖")
st.title("🤖 My Chatbot 🤖")

# Function to handle file upload
def upload_file(files):
    responses = []

    for file in files:
        if str(file) not in save_path:
            data = file.read()

            response = requests.post(
                "http://localhost:8000/upload-file", files={"file": (file.name, data)}
            )
            responses.append(response.json())
        else:
            pass
    if len(responses) > 1:
        st.markdown("Files uploaded and vectorized!")
    else:
        st.markdown("File uploaded and vectorized!")
    return responses

# File upload section
uploaded_files = st.file_uploader("Upload a document to ask specific questions", type=["txt", "pdf"], accept_multiple_files=True)
if uploaded_files:
    upload_file(uploaded_files)

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
    result = response_data.get("result", "Error in response")
    sources = response_data.get("sources", [])
    return result, sources


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

    if query_type == 'General':
        endpoint = "http://127.0.0.1:8000/qa-request"
    else:
        endpoint = "http://127.0.0.1:8000/qa-documents"

    with st.chat_message("AI"):
        ai_response = get_response(user_query, st.session_state.chat_history, endpoint)
        result = ai_response[0]
        sources = ai_response[1]

        st.markdown(result)
        if sources:
            st.markdown("**Sources:**")
        processed_sources = set()
        for source in sources:
            file_name = os.path.basename(source)
            if file_name not in processed_sources:
                st.markdown(file_name)
                processed_sources.add(file_name)

        # Append the AIMessage to the session state chat history
        st.session_state.chat_history.append(AIMessage(content=result))
