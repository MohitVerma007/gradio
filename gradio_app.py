import gradio as gr
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Get the Hugging Face token from the environment variables
hf_token = os.getenv("HUGGINGFACE_TOKEN")

# Initialize the InferenceClient with the model and token
client = InferenceClient("microsoft/Phi-3-mini-4k-instruct", token=hf_token)

# Response generation function
def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

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

# Gradio Interface
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)"),
    ],
)

# Launch Gradio interface
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
