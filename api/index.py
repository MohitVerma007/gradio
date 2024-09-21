from fastapi import FastAPI, Request
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Get the Hugging Face token from the environment variables
hf_token = os.getenv("HUGGINGFACE_TOKEN")

# Initialize the InferenceClient with the model and token
client = InferenceClient("microsoft/Phi-3-mini-4k-instruct", token=hf_token)

# Create FastAPI app
app = FastAPI()

# Response generation function
def respond(message, history: list[tuple[str, str]], system_message, max_tokens, temperature, top_p):
    messages = [{"role": "system", "content": system_message}]
    
    # Add the conversation history to the message list
    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    # Add the new user message to the message list
    messages.append({"role": "user", "content": message})

    response = ""

    # Collect tokens from the model
    for msg in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = msg.choices[0].delta.content
        if token:
            response += token

    return response  # Return the complete response directly

# Endpoint to access the API via FastAPI
@app.post("/generate-response/")
async def generate_response(request: Request):
    body = await request.json()
    message = body["message"]
    history = body.get("history", [])
    system_message = body.get("system_message", "You are a friendly Chatbot.")
    max_tokens = body.get("max_tokens", 512)
    temperature = body.get("temperature", 0.7)
    top_p = body.get("top_p", 0.95)

    # Call respond function and return the response
    response = respond(message, history, system_message, max_tokens, temperature, top_p)
    return {"response": response}
