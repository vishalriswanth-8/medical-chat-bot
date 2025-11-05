# This script handles the one-time process of creating and saving the vector database.
# You will need to install the necessary libraries.
#
# If you have a GPU, run `pip install faiss-gpu`.
# If you do not have a GPU, run `pip install faiss-cpu`.  <-- This is the command you need.
# pip install sentence-transformers
# pip install langchain-community

import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 1. LOAD YOUR MEDICAL DATA ---
# In a real-world scenario, you would load data from files (PDFs, text, etc.).
# This is a simplified example with a list of strings.
medical_data = [
    "Aspirin, a nonsteroidal anti-inflammatory drug (NSAID), is used to treat pain, fever, and inflammation. It can also be used as an antiplatelet agent to prevent blood clots.",
    "The pancreas is an organ located in the abdomen. It plays a crucial role in converting the food we eat into fuel for the body's cells. It has two main functions: an exocrine function that helps in digestion and an endocrine function that regulates blood sugar.",
    "Hypertension, or high blood pressure, is a condition where the force of the blood against the artery walls is too high. It can lead to serious health problems, including heart disease and stroke. It can be managed with lifestyle changes and medication.",
    "Diabetes mellitus is a chronic condition that affects how your body turns food into energy. It is characterized by high blood glucose (sugar) levels. There are two main types: Type 1 and Type 2."
]

# --- 2. SPLIT DOCUMENTS INTO CHUNKS ---
# This is important for handling long documents.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.create_documents(medical_data)

# --- 3. CREATE THE EMBEDDINGS AND VECTOR STORE ---
print("Creating vector store...")
# Make sure to use an embedding-specific model like `nomic-embed-text`.
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = FAISS.from_documents(documents, embeddings)
print("Vector store created successfully.")

# --- 4. SAVE THE VECTOR STORE TO A FILE ---
db_path = "faiss_index.faiss"
vector_store.save_local(db_path)
print(f"Vector database saved to '{os.path.abspath(db_path)}'")
