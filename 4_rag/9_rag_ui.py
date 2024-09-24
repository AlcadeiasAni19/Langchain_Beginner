import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from pdfminer.high_level import extract_text

load_dotenv()
# Define the fixed file and persistent directory for the vector store
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "iphone_user_guide.pdf")  # Keep this file fixed
persistent_directory = os.path.join(current_dir, "db", "chroma_db_ui")


def initialize_vector_store(file_path, persist_directory):

    # Check if vector store already exists and delete it if needed
    if os.path.exists(persist_directory):
        print("Existing vector store found. Deleting...")
        shutil.rmtree(persist_directory)

    print("Initializing vector store...")

    # Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path.")

    pdf_text = extract_text(file_path)
    temp_text_file = os.path.join(current_dir, "temp_text.txt")
    with open(temp_text_file, 'w', encoding='utf-8', errors='ignore') as f:
        f.write(pdf_text)
    # Read the text content from the file
    # loader = TextLoader(file_path)
    # documents = loader.load()
    documents = Document(page_content=pdf_text)
    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents([documents])

    # Create embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # Use OpenAI embedding model

    # Create the vector store and persist it
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    db.persist()

    return db


def ask_question(vector_store, question):
    """Takes a question and retrieves the most relevant document chunks from the vector store."""

    # Initialize the OpenAI model
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Create a retrieval-based QA chain
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    response = qa_chain.run(question)
    return response


# Streamlit UI
st.title("Document Q&A with AI")

# Initialize the vector store with a fixed file
if 'vector_store' not in st.session_state:
    vector_store = initialize_vector_store(file_path, persistent_directory)
    st.session_state['vector_store'] = vector_store

# Allow the user to ask a question
user_question = st.text_input("Ask:")

if st.button("Get Answer"):
    if user_question:
        answer = ask_question(st.session_state['vector_store'], user_question)
        st.write(f"Answer: {answer}")
    else:
        st.write("Please enter a question.")
