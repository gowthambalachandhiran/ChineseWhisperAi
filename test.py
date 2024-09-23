import openai  # For ChatGPT
import streamlit as st
import random
import time  # To handle the delay for Gemini
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI  # For Gemini API
from groq import Groq

# Llama Model Initialization
class LlamaModel:
    def __init__(self, client, temperature, max_tokens):
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, natural_query):
        try:
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": natural_query}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=1,
                stream=False,
                stop=None,
            )
            return response.choices[0].message.content  # Properly extract content
        except Exception as e:
            print(f"Llama Model Error: {e}")
            return "An error occurred with Llama model."

# ChatGPT Model Initialization
class ChatGPTModel:
    def __init__(self, temperature, max_tokens):
        openai.api_key = st.secrets["OPENAI_API_KEY"]  # Fetch API key from Streamlit secrets
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            print(f"ChatGPT Error: {e}")
            return "An error occurred with ChatGPT."

# Gemini Model Initialization
class GeminiModel:
    def __init__(self, temperature, max_tokens):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=temperature,
            max_output_tokens=max_tokens,
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )

    def generate(self, prompt):
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"Gemini Model Error: {e}")
            return "An error occurred with Gemini."

# Sidebar to adjust parameters
st.sidebar.title("Model Parameters")

# ChatGPT sliders
chatgpt_temperature = st.sidebar.slider("ChatGPT Temperature", min_value=0.0, max_value=1.0, value=0.7)
chatgpt_max_tokens = st.sidebar.slider("ChatGPT Max Tokens", min_value=100, max_value=1024, value=500)

# Gemini sliders
gemini_temperature = st.sidebar.slider("Gemini Temperature", min_value=0.0, max_value=1.0, value=0.7)
gemini_max_tokens = st.sidebar.slider("Gemini Max Tokens", min_value=100, max_value=1024, value=500)

# Llama sliders
llama_temperature = st.sidebar.slider("Llama Temperature", min_value=0.0, max_value=1.0, value=0.7)
llama_max_tokens = st.sidebar.slider("Llama Max Tokens", min_value=100, max_value=1024, value=500)

# Initialize the models
chatgpt = ChatGPTModel(chatgpt_temperature, chatgpt_max_tokens)
gemini = GeminiModel(gemini_temperature, gemini_max_tokens)
groq_api_key = st.secrets["groq_api_key"]

# Initialize Groq client
client = Groq(api_key=groq_api_key)

# Initialize Llama
llama = LlamaModel(client, llama_temperature, llama_max_tokens)

# Mapping model functions
model_map = {
    "ChatGPT": lambda input_text: chatgpt.generate(input_text),
    "Gemini": lambda input_text: gemini.generate(input_text),
    "Llama": lambda input_text: llama.generate(input_text)
}

# Function to randomly select a model
def random_model():
    return random.choice(list(model_map.keys()))

# Function to run Chinese Whisper across models
def run_chinese_whisper(start_topic, iterations):
    history = []
    current_topic = start_topic
    for i in range(iterations):
        # Select a random model
        model = random_model()
        if model == "Gemini":
            time.sleep(2)  # Delay for Gemini

        # Generate response based on the current topic
        response = model_map[model](f"Generate a detailed response to this topic: '{current_topic}'")
        
        history.append({
            "Iteration": i + 1,
            "Model": model,
            "Input Topic": current_topic,
            "Response": response
        })

        # Select another random model for inference and next response
        next_model = random_model()
        if next_model == "Gemini":
            time.sleep(2)  # Delay for Gemini

        # Infer new topic and generate next response
        infer_and_respond = model_map[next_model](f"First, infer a brief topic (max 4 words) from this text: '{response}'. Then, generate a detailed response to your inferred topic. Format your answer as 'Inferred Topic: [your inferred topic]. Response: [your response]'")
        
        # Split the inferred topic and response, handling potential formatting issues
        try:
            if "Inferred Topic:" in infer_and_respond and "Response:" in infer_and_respond:
                parts = infer_and_respond.split("Response:", 1)
                inferred_topic = parts[0].replace("Inferred Topic:", "").strip()
                next_response = parts[1].strip()
            else:
                # If the expected format is not found, make a best guess
                words = infer_and_respond.split()
                inferred_topic = " ".join(words[:4])  # Take first 4 words as the inferred topic
                next_response = " ".join(words[4:])  # Rest is the response
        except Exception as e:
            print(f"Error processing model response: {e}")
            inferred_topic = "Error in topic inference"
            next_response = "Error in response generation"

        history.append({
            "Iteration": i + 1,
            "Model": next_model,
            "Input Topic": f"Inferred from previous: {inferred_topic}",
            "Response": next_response
        })

        # Update the current topic for the next iteration
        current_topic = inferred_topic

    return history

# Streamlit UI
st.title("Chinese Whisper with AI Models")

# User inputs
start_topic = st.text_input("Enter the starting topic:")
iterations = st.number_input("Number of iterations:", min_value=1, value=5)

# Start button logic
if st.button("Start"):
    if start_topic:
        history = run_chinese_whisper(start_topic, iterations)
        st.subheader("Topic Evolution:")
        for item in history:
            st.write(f"Iteration {item['Iteration']}:")
            st.write(f"Model: {item['Model']}")
            st.write(f"Input Topic: {item['Input Topic']}")
            st.write(f"Response: {item['Response']}")
    else:
        st.error("Please enter a starting topic.")
