# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 17:00:19 2024

@author: gowtham.balachan
"""

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
            return response.choices[0].message.content
        except Exception as e:
            print(f"Llama Model Error: {e}")
            return "An error occurred with Llama model."
