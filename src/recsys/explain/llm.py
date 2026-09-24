import os
from src.recsys.domain import Recommendation
from typing import List

class LLMExplainer:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                self.client = None
        else:
            self.client = None
            
    def generate_explanation(self, user_history_names: List[str], product_name: str, category: str, breakdown) -> str:
        if not user_history_names:
            return f"This {category} is highly popular among our shoppers right now."
            
        history_str = ", ".join(user_history_names[-3:])
        
        if self.client:
            prompt = f"User recently bought/viewed: {history_str}. We are recommending: {product_name} ({category}). Write a one-sentence personalized explanation why this is a good recommendation. Keep it friendly and concise."
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                pass
                
        if breakdown:
            if breakdown.cf > breakdown.content:
                return f"Shoppers who bought items like '{user_history_names[-1]}' also frequently purchased the {product_name}."
            else:
                return f"Because you showed interest in '{user_history_names[-1]}', we think this {category} is a perfect match."
                
        return f"Based on your recent activity, we highly recommend the {product_name}."
