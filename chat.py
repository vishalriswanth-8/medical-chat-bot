import requests
import json
import os

# --- Configuration ---
API_KEY = "123"
SERVER_URL = "http://127.0.0.1:5000/chat"

# --- Main Program ---
def main():
    # Prompt the user to enter their unique ID
    print("Hello! I'm your health assistant.")
    user_id = input("Please enter your user ID: ")
    print("--------------------------------------------------")
    print("Type 'exit' to end the conversation.")
    print("--------------------------------------------------")

    # New: Conversation history list
    conversation_history = []
    
    def get_chatbot_response(message):
        """Sends a message and conversation history to the Flask API."""
        
        conversation_history.append({"role": "user", "content": message})
        
        payload = {
            "api_key": API_KEY,
            "user_id": user_id,  # Now uses the dynamic user_id
            "message": message,
            "history": conversation_history
        }
        
        try:
            response = requests.post(SERVER_URL, json=payload)
            response.raise_for_status()
            chatbot_response = response.json().get("response", "An error occurred with the response.")
            
            conversation_history.append({"role": "assistant", "content": chatbot_response})
            
            return chatbot_response
        except requests.exceptions.RequestException as e:
            return f"Error connecting to the chatbot server: {e}"

    # --- Continuous Chat Loop ---
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye! Stay healthy! 👋")
            break
        
        chatbot_response = get_chatbot_response(user_input)
        print(f"Chatbot: {chatbot_response}")

if __name__ == "__main__":
    main()