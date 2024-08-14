import os

import chromadb
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain.docstore.document import Document

# Define the directory containing the pdf file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "iphone_user_guide.pdf")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")
pdf_text = extract_text(file_path)

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )
    
    temp_text_file = os.path.join(current_dir, "temp_text.txt")
    with open(temp_text_file, 'w', encoding='utf-8', errors='ignore') as f:
        f.write(pdf_text)

    # Read the text content from the file
    # loader = TextLoader(file_path, encoding="utf-8")
    # documents = loader.load()

    # Create a Document object
    document = Document(page_content=pdf_text)

    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents([document])

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Display information about the split documents
    # print("\n--- Document Chunks Information ---")
    # print(f"Number of document chunks: {len(docs)}")
    # print(f"Sample chunk:\n{docs[0].page_content}\n")

    # Create embeddings
    # print("\n--- Creating embeddings ---")
    # embeddings = OpenAIEmbeddings(
    #     model="text-embedding-3-small"
    #)  # Update to a valid embedding model if needed
    #print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")

    os.remove(temp_text_file)
    # client = chromadb.Client()
    # collection = client.create_collection(name="pdf-docs")

    # vectors = [embedding.tolist() for embedding in embeddings]
    # metadata = [{"text": chunk} for chunk in chunks]

    # # Insert embeddings into the collection
    # collection.add(vectors=vectors, metadata=metadata)
else:
    print("Vector store already exists. No need to initialize.")

# query = "Who is Menelaus’s wife?"
# docs = db.similarity_search(query)

# # print results
# print(docs[0].page_content)
