import os
import json
import requests
import re
from flask import Flask, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS

# Import for the vector database
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import for image processing
from PIL import Image
import pytesseract

# For Windows, replace with your install path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- Configuration --- app
API_KEY = "123"
FAISS_DB_PATH = "faiss_index.faiss"
PERSONAL_DB_DIR = "personal_vector_dbs"
# Global variables
public_vector_store = None
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# --- Vector Database Loading ---
def init_public_vector_db():
    """Initializes and loads the public FAISS vector database."""
    global public_vector_store
    
    if not os.path.exists(FAISS_DB_PATH):
        print(f"Error: Public vector database file '{FAISS_DB_PATH}' not found.")
        print("Please run the one-time database creation script first.")
        return False
    
    print("Loading public vector database...")
    public_vector_store = FAISS.load_local(FAISS_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    print("Public vector database ready.")
    return True

def load_or_create_personal_db(user_id: str) -> FAISS:
    """
    Loads a user's personal vector database, or creates a new one if it doesn't exist.
    """
    user_db_path = os.path.join(PERSONAL_DB_DIR, f"user_{user_id}")
    
    if os.path.exists(user_db_path):
        print(f"Loading personal vector database for user: {user_id}")
        return FAISS.load_local(user_db_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"Creating new personal vector database for user: {user_id}")
        if not os.path.exists(PERSONAL_DB_DIR):
            os.makedirs(PERSONAL_DB_DIR)
        
        placeholder_text = ["This is the start of your personal health log."]
        db = FAISS.from_texts(placeholder_text, embeddings)
        db.save_local(user_db_path)
        return db

def update_personal_db(user_id: str, message: str):
    """
    Updates a user's personal vector database with a new message.
    """
    try:
        user_db_path = os.path.join(PERSONAL_DB_DIR, f"user_{user_id}")
        
        personal_db = load_or_create_personal_db(user_id)

        # Create a unique ID for this entry (e.g., a timestamp)
        unique_id = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        metadata = {"source": f"chat_log_{unique_id}"}
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(message)
        
        personal_db.add_texts(texts, metadatas=[metadata] * len(texts))
        personal_db.save_local(user_db_path)
        print(f"Personal vector database for user '{user_id}' updated with new message.")
    except Exception as e:
        print(f"Error updating personal database: {e}")

def delete_personal_db_entry(user_id: str, entry_to_delete: str):
    """
    Deletes an entry from a user's personal vector database by rebuilding the index.
    """
    try:
        user_db_path = os.path.join(PERSONAL_DB_DIR, f"user_{user_id}")
        if not os.path.exists(user_db_path):
            print(f"Personal database for user '{user_id}' not found. Nothing to delete.")
            return

        personal_db = FAISS.load_local(user_db_path, embeddings, allow_dangerous_deserialization=True)
        
        all_docs = list(personal_db.docstore._dict.values())
        
        docs_to_keep = [doc for doc in all_docs if entry_to_delete.lower() not in doc.page_content.lower()]
        
        if len(docs_to_keep) == len(all_docs):
            print(f"No matching entry found to delete for user '{user_id}'.")
            return
            
        print(f"Deleting entry for user '{user_id}'. {len(all_docs) - len(docs_to_keep)} document(s) removed.")
        
        new_db = FAISS.from_documents(docs_to_keep, embeddings)
        new_db.save_local(user_db_path)
        print(f"Personal database for user '{user_id}' rebuilt successfully.")
        
    except Exception as e:
        print(f"Error deleting entry: {e}")

# --- LLM Integration (RAG) ---
def get_llm_response(message: str, user_id: str, conversation_history: list) -> str:
    """
    Calls the local LLM to get a RAG-augmented response using both public
    and personal user data.
    """
    try:
        # Step 1: Retrieve relevant context from both databases
        personal_db = load_or_create_personal_db(user_id)
        
        public_docs = public_vector_store.similarity_search(message, k=2)
        # When a user asks about their health, retrieve more comprehensive info
        if "my health" in message.lower() or "my condition" in message.lower():
            personal_docs = personal_db.similarity_search(message, k=10)
        else:
            personal_docs = personal_db.similarity_search(message, k=3)
        
        all_docs = public_docs + personal_docs
        context = "\n\n".join([doc.page_content for doc in all_docs])
        
        # Format the history for the prompt
        history_prompt = "\n".join([
            f"{entry['role'].capitalize()}: {entry['content']}" 
            for entry in conversation_history
        ])

        # Step 2: Augment the prompt with the retrieved context
        augmented_prompt = f"""
        You are MediBot, a kind and helpful healthcare assistant. Your main goal is to help common village people with simple health questions.
        
        **Your responses must be:**
        * **Simple and Easy to Understand:** Use simple words and short sentences. Avoid complex medical terms.
        * **Empathetic and Emotional:** Show you care. Use a friendly tone and add a single, relevant emoji at the end of each response.
        * **Investigative:** Ask clarifying questions to understand the user's condition better.

        Here is the history of the conversation so far:
        <conversation_history>
        {history_prompt}
        </conversation_history>

        Use the following medical context to answer the user's question. The context includes both
        general health information and the user's personal health notes and conversation history.
        
        **Important:** When the user shares personal information (like their name or a health issue), do not say "I saved this" or "I have added this to your health log." Just respond naturally and kindly, showing you understood. For example, if they say "i have back pain," you could reply, "I'm sorry to hear about your back pain. That sounds really tough. Is this a new pain? 😔"
        
        If the context does not contain the answer, state that you don't have enough information in a kind way.
        Don't try to make up an answer. You can only answer medical questions.
        
        <context>
        {context}
        </context>
        
        The user's question is: {message}
        """

        # Step 3: Call the LLM with the augmented prompt
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma3:4b",
                "prompt": augmented_prompt
            },
            stream=True
        )

        full_response = ""
        for line in response.iter_lines():
            if line:
                json_chunk = json.loads(line)
                full_response += json_chunk.get("response", "")
        
        return full_response.strip()

    except requests.exceptions.ConnectionError:
        return "Sorry, I can't connect to the local LLM server. Please make sure Ollama is running. 😞"
    except Exception as e:
        return f"An error occurred: {e} 😕"

# --- Scheduler Tasks ---
def send_proactive_tips():
    """Simulates sending health tips to users."""
    print(f"[{datetime.now()}] PROACTIVE TIP: Remember to drink plenty of water throughout the day! 💧")

def check_reminders():
    """Simulates checking and sending medication/vaccine reminders."""
    print(f"[{datetime.now()}] REMINDER: Check the database for upcoming medication and vaccine schedules to send to users. ⏰")

# --- Flask App ---
app = Flask(__name__)
CORS(app) # Add this line to enable CORS for all routes

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    
    if data.get("api_key") != API_KEY:
        return jsonify({"error": "Invalid API key"}), 401
    
    user_id = data.get("user_id")
    message = data.get("message")
    conversation_history = data.get("history", [])
    
    if not user_id or not message:
        return jsonify({"error": "Missing user_id or message"}), 400
    
    # --- New Logic for Selective Storage and Deletion ---
    store_keywords = ["remember", "keep in mind", "save this", "store this", "my name is", "i have", "i am"]
    delete_keywords = ["forget", "remove", "don't have", "not have"]
    
    message_lower = message.lower()
    
    # Check for a deletion command at the start of the message
    if any(message_lower.startswith(keyword) for keyword in delete_keywords):
        content_to_delete = re.sub(r'|'.join(delete_keywords), '', message_lower, flags=re.IGNORECASE).strip()
        if content_to_delete:
            delete_personal_db_entry(user_id, content_to_delete)
            # The LLM will generate the final response
            return jsonify({"response": get_llm_response(message, user_id, conversation_history)})
    
    # Check for a storage command at the start of the message
    if any(message_lower.startswith(keyword) for keyword in store_keywords):
        content_to_store = re.sub(r'|'.join(store_keywords), '', message_lower, flags=re.IGNORECASE).strip()
        if content_to_store:
            update_personal_db(user_id, content_to_store)
            # The LLM will generate the final response, which is the key change
            return jsonify({"response": get_llm_response(message, user_id, conversation_history)})

    # If no specific command is found, proceed with normal chat
    response = get_llm_response(message, user_id, conversation_history)
    
    return jsonify({"response": response})

@app.route('/save_medical_info', methods=['POST'])
def save_info():
    data = request.get_json()
    
    if data.get("api_key") != API_KEY:
        return jsonify({"error": "Invalid API key"}), 401
    
    user_id = data.get("user_id")
    medical_info = data.get("medical_info")
    
    if not user_id or not medical_info:
        return jsonify({"error": "Missing user_id or medical_info"}), 400

    update_personal_db(user_id, medical_info)
    return jsonify({"message": "Medical info saved successfully to personal vector database."})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    data = request.get_json()

    if data.get("api_key") != API_KEY:
        return jsonify({"error": "Invalid API key"}), 401

    image_path = data.get("image_path")
    user_id = data.get("user_id")
    
    if not image_path or not user_id:
        return jsonify({"error": "Missing image_path or user_id"}), 400

    try:
        with Image.open(image_path) as img:
            extracted_text = pytesseract.image_to_string(img)
            
        if not extracted_text.strip():
            return jsonify({"response": "No readable text found in the image."})

        update_personal_db(user_id, extracted_text)

        response = get_llm_response(extracted_text, user_id, [])

        return jsonify({"response": response, "extracted_text": extracted_text})

    except FileNotFoundError:
        return jsonify({"error": "Image file not found at the specified path."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Main Application ---
if __name__ == '__main__':
    if not init_public_vector_db():
        exit()

    if not os.path.exists(PERSONAL_DB_DIR):
        os.makedirs(PERSONAL_DB_DIR)

    scheduler = BackgroundScheduler()
    scheduler.add_job(send_proactive_tips, 'interval', minutes=30)
    scheduler.add_job(check_reminders, 'interval', minutes=60)
    scheduler.start()

    app.run(host='0.0.0.0', port=5000)