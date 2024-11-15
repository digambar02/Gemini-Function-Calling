import os
import requests
import google.generativeai as genai
import gradio as gr
from google.api_core import retry


def body_part_list() -> list[str]:
    """
    This function returns the body parts for which exercise is available.
    """
    url = "https://exercisedb.p.rapidapi.com/exercises/bodyPartList"
    
    headers = {
        "x-rapidapi-key": os.getenv('RAPIDAPI_KEY'),
        "x-rapidapi-host": "exercisedb.p.rapidapi.com"
    }
    
    response = requests.get(url, headers=headers)
    return response.json()

def exercise(body_part: str) -> dict():
    """
    Body part is used as input to the function. 
    Returns a dictionary of exercise details.
    """
    url = f"https://exercisedb.p.rapidapi.com/exercises/bodyPart/{body_part}"
    
    querystring = {"limit":"10","offset":"0"}
    
    headers = {
        "x-rapidapi-key": os.getenv('RAPIDAPI_KEY'),
        "x-rapidapi-host": "exercisedb.p.rapidapi.com"
    }
    
    response = requests.get(url, headers=headers, params=querystring)
    
    return {
        'bodyPart': response.json()[0]['bodyPart'],
        'muscle_name': response.json()[0]['target'],
        'exercise_name': response.json()[0]['name'],
        'instructions': response.json()[0]['instructions'],
        'gifUrl': response.json()[0]['gifUrl']
    }

def initialize_model():
    """Initialize the Gemini model with exercise functions"""
    GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # Define the tools (functions) available to the model
    db_tools = [body_part_list, exercise]
    
    # System instruction for the fitness trainer role
    instruction = """You are a helpful fitness trainer chatbot. 
    The user will provide you with the body part they are interested in training. 
    Use the body_part_list function to match the body part the user has provided. 
    If the body part is available in the list provided by the body_part_list function proceed, 
    otherwise respond by giving list of body parts you can assist with. 
    After getting expected body part, use exercise function and use body part as input to the function. 
    Extract body part, muscle name, exercise name and exercise instructions. Present in readable format"""
    
    model = genai.GenerativeModel(
        "gemini-1.5-flash-latest",
        tools=db_tools,
        system_instruction=instruction
    )
    
    return model

def create_chat():
    """Create a new chat instance with automatic function calling"""
    model = initialize_model()
    chat = model.start_chat(enable_automatic_function_calling=True)
    return chat

def respond(message, chat_history):
    """Generate response using the fitness trainer chatbot"""
    try:
        # Initialize chat if first message
        if not hasattr(respond, 'chat'):
            respond.chat = create_chat()
        
        # Define retry policy
        retry_policy = {"retry": retry.Retry(predicate=retry.if_transient_error)}
        
        # Get response from model
        response = respond.chat.send_message(message, request_options=retry_policy)
        
        # Add to chat history
        chat_history.append((message, response.text))
        
        return "", chat_history
    except Exception as e:
        return "", chat_history + [("Error", f"An error occurred: {str(e)}")]

def clear_chat():
    """Reset chat history and create new chat instance"""
    if hasattr(respond, 'chat'):
        delattr(respond, 'chat')
    return None, []

def create_gradio_app():
    """Create and configure the Gradio app interface"""
    with gr.Blocks(css="footer {visibility: hidden}") as demo:
        gr.Markdown("""
        # 🏋️‍♂️ AI Fitness Trainer
        Chat with your personal AI fitness trainer! Tell me which body part you want to train, 
        and I'll provide you with specific exercises and instructions.
        """)
        
        chatbot = gr.Chatbot(
            height=600,
            show_label=False,
            container=True,
            bubble_full_width=False,
        )
        
        with gr.Row():
            txt = gr.Textbox(
                scale=4,
                show_label=False,
                placeholder="Example: I want to train chest",
                container=False,
            )
            submit_btn = gr.Button("Send", scale=1, variant="primary")
        
        clear_btn = gr.Button("Clear Chat")
        
        # Event handlers
        submit_btn.click(
            fn=respond,
            inputs=[txt, chatbot],
            outputs=[txt, chatbot],
        )
        txt.submit(
            fn=respond,
            inputs=[txt, chatbot],
            outputs=[txt, chatbot],
        )
        clear_btn.click(
            fn=clear_chat,
            inputs=[],
            outputs=[txt, chatbot],
        )
        
    return demo

if __name__ == "__main__":
    # Create and launch the app
    demo = create_gradio_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )