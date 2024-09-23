# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 16:59:43 2024

@author: gowtham.balachan
"""

import openai
import streamlit as st

class ChatGPTModel:
    def __init__(self, temperature, max_tokens):
        openai.api_key = st.secrets["OPENAI_API_KEY"]
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
