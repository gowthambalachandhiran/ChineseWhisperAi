# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 17:00:00 2024

@author: gowtham.balachan
"""

from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
import streamlit as st

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
