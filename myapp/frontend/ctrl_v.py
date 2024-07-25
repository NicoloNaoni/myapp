import streamlit as st
import requests
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="My Chatbot", page_icon="🤖")
st.title("My Chatbot")

# Initialize session state for chat history if not already done
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


# Function to handle file upload
def upload_file(file):
    data = file.read()
    response = requests.post(
        "http://localhost:8000/upload-file", files={"file": data}
    )
    if response.status_code == 200:
        st.success("File uploaded and vectorized successfully!")
    else:
        st.error(f"Error: {response.json().get('error', 'Unknown error')}")


# File upload section
uploaded_file = st.file_uploader("Upload a document to ask specific questions", type=["txt", "pdf"])
if uploaded_file is not None:
    upload_file(uploaded_file)


def get_response(query, endpoint):
    payload = {
        "query": query,
        "history": st.session_state.chat_history  # Send chat history with the request
    }
    response = requests.post(endpoint, json=payload)
    if response.status_code == 200:
        response_data = response.json()
        print(response_data)  # Print full response for debugging
        return response_data.get("result", "Error in response")
    else:
        st.error(f"Error: {response.json().get('error', 'Unknown error')}")
        return "Error in response"


# Toggle for query type
query_type = st.radio("Choose query type:", ('General', 'Document-Specific'))

# Display chat history
for message in st.session_state.chat_history:
    if "user" in message:
        with st.chat_message("Human"):
            st.markdown(message["user"])
    if "bot" in message:
        with st.chat_message("AI"):
            st.markdown(message["bot"])

# Single chat input
user_query = st.chat_input("Your message")
if user_query:
    with st.chat_message("Human"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"user": user_query})  # Save user's message to history

    endpoint = "http://127.0.0.1:8000/qa-request" if query_type == 'General' else "http://127.0.0.1:8000/qa-documents"
    ai_response = get_response(user_query, endpoint)

    # Ensure ai_response is a string
    if not isinstance(ai_response, str):
        ai_response = str(ai_response)

    # Print ai_response to debug
    print("AI Response:", ai_response)

    with st.chat_message("AI"):
        st.markdown(ai_response)  # Directly display the answer
    st.session_state.chat_history.append({"bot": ai_response})  # Save AI's response to history