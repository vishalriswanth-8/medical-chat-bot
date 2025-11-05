import requests
import json

# --- Configuration ---
API_KEY = "123"  # Must match the API key in your Flask app
USER_ID = "user123" # A unique user ID for the session
SERVER_URL = "http://127.0.0.1:5000/chat"

print("Hello! I'm your health assistant. Type 'exit' to end the conversation.")
print("-" * 50)

def get_chatbot_response(message):
    """Sends a message to the Flask API and returns the chatbot's response."""
    payload = {
        "api_key": API_KEY,
        "user_id": USER_ID,
        "message": message
    }
    
    try:
        response = requests.post(SERVER_URL, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        return response.json().get("response", "An error occurred with the response.")
    except requests.exceptions.RequestException as e:
        return f"Error connecting to the chatbot server: {e}"

# --- Continuous Chat Loop ---
while True:
    user_input = input("You: ")
    
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Chatbot: Goodbye! Stay healthy!")
        break
    
    # Get and print the chatbot's response
    chatbot_response = get_chatbot_response(user_input)
    print(f"Chatbot: {chatbot_response}")