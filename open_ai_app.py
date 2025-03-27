import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Chat with the Streamlit docs, powered by LlamaIndex",
    page_icon="🦙",
    layout="centered",
)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API Key is missing. Check your .env file.")
    st.stop()

st.title("Chat with the Custom docs, powered by LlamaIndex 💬🦙")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about your custom data!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    try:
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        llm = OpenAI(
            model="gpt-3.5-turbo",
            temperature=0.2,
            system_prompt="""You are a highly specialized data analysis and insights bot designed to process, analyze, and interact with structured data such as resume pdf files. Your primary purpose is to assist users by answering questions about their data, and providing actionable insights based on their datasets.""",
        )
        index = VectorStoreIndex.from_documents(docs, llm=llm)
        return index
    except Exception as e:
        st.error(f"Error loading data: {e}")
        raise

try:
    index = load_data()
except Exception:
    st.stop()

if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True, streaming=True
    )

if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        try:
            # Collect response chunks
            response_stream = st.session_state.chat_engine.stream_chat(
                st.session_state.messages[-1]["content"]
            )
            response_content = "".join(response_stream.response_gen)
            
            # Display the final response at the end
            st.write(response_content)
            
            # Add the final response to session state
            st.session_state.messages.append(
                {"role": "assistant", "content": response_content}
            )
        except Exception as e:
            st.error(f"Error generating response: {e}")

