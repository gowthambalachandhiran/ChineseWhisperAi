# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 17:00:54 2024

@author: gowtham.balachan
"""

import streamlit as st

def sidebar_parameters():
    # ChatGPT sliders
    chatgpt_temperature = st.sidebar.slider("ChatGPT Temperature", min_value=0.0, max_value=1.0, value=0.7)
    chatgpt_max_tokens = st.sidebar.slider("ChatGPT Max Tokens", min_value=100, max_value=1024, value=500)

    # Gemini sliders
    gemini_temperature = st.sidebar.slider("Gemini Temperature", min_value=0.0, max_value=1.0, value=0.7)
    gemini_max_tokens = st.sidebar.slider("Gemini Max Tokens", min_value=100, max_value=1024, value=500)

    # Llama sliders
    llama_temperature = st.sidebar.slider("Llama Temperature", min_value=0.0, max_value=1.0, value=0.7)
    llama_max_tokens = st.sidebar.slider("Llama Max Tokens", min_value=100, max_value=1024, value=500)

    return chatgpt_temperature, chatgpt_max_tokens, gemini_temperature, gemini_max_tokens, llama_temperature, llama_max_tokens
