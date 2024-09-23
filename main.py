import streamlit as st
import random
import time
from models.chatgpt_model import ChatGPTModel
from models.gemini_model import GeminiModel
from models.llama_model import LlamaModel
from ui.ui import sidebar_parameters
from groq import Groq
import pandas as pd  # For table display

# Sidebar for model parameters
(chatgpt_temp, chatgpt_tokens, gemini_temp, gemini_tokens, llama_temp, llama_tokens) = sidebar_parameters()

# Initialize the models
chatgpt = ChatGPTModel(chatgpt_temp, chatgpt_tokens)
gemini = GeminiModel(gemini_temp, gemini_tokens)
groq_api_key = st.secrets["groq_api_key"]

# Initialize Groq client
client = Groq(api_key=groq_api_key)

# Initialize Llama
llama = LlamaModel(client, llama_temp, llama_tokens)

# Mapping model functions
model_map = {
    "ChatGPT": lambda input_text: chatgpt.generate(input_text),
    "Gemini": lambda input_text: gemini.generate(input_text),
    "Llama": lambda input_text: llama.generate(input_text)
}

# Color mapping for each model
color_map = {
    "ChatGPT": "blue",
    "Gemini": "green",
    "Llama": "purple"
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

        # Display response in the model-specific color
        st.markdown(f"<span style='color:{color_map[model]}; font-weight:bold;'>Iteration {i+1} | {model}'s response:</span> {response}", unsafe_allow_html=True)

        # Select another random model for inference and next response
        next_model = random_model()
        if next_model == "Gemini":
            time.sleep(2)  # Delay for Gemini

        # Infer new topic and generate next response
        infer_and_respond = model_map[next_model](f"First, infer a brief topic (max 4 words) from this text: '{response}'. Then, generate a detailed response to your inferred topic.")
        
        # Split the inferred topic and response
        try:
            words = infer_and_respond.split()

            # Limit inferred topic to first 4 words (max)
            inferred_topic = " ".join(words[:30])  # Take up to first 4 words

            # Limit next response to a maximum of 10 words
            next_response = " ".join(words[:150])  # Take max 10 words

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

        # Display next response in the model-specific color
        st.markdown(f"<span style='color:{color_map[next_model]}; font-weight:bold;'>Iteration {i+1} | {next_model}'s response:</span> {next_response}", unsafe_allow_html=True)

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
        
        # Prepare a list of all topics discussed
        topics_data = {
            "Iteration": [],
            "Model": [],
            "Input Topic": []
        }
        
        for item in history:
            topics_data["Iteration"].append(item["Iteration"])
            topics_data["Model"].append(item["Model"])
            topics_data["Input Topic"].append(item["Input Topic"])
        
        # Display the topics in a table
        df = pd.DataFrame(topics_data)
        st.table(df)
    else:
        st.error("Please enter a starting topic.")
