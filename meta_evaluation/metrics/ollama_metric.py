import torch
import requests, json

class OLLAMA_metric:

    def __init__(self, prompt):
        self.name = "OLLAMA-Eval " + prompt 
        self.prompt = open(prompt).read().strip()
        self.url = "http://localhost:11434/api/chat"

    def llama3_response(self, prompt):
        data = {
            "model": "llama3:70b",
            "messages": [
                 {"role": "system", "content": "You are assistant trying to help the user!"},
                 {"role": "user", "content": prompt}
            ],
            "stream": False
        }
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        response = requests.post(self.url, headers=headers, json=data)
        return response.json()['message']['content']
    
    def compute_metric(self, complex, simplified, references):

        scores = []
        for index, (complex_single, simp_single) in enumerate(zip(complex, simplified)):
            print("Instance ", index, len(complex))
            prompt = self.prompt.replace("||complex||", complex_single)
            prompt = prompt.replace("||simplification||", simp_single)
            score = int(self.llama3_response(prompt).strip())
            scores.append(score)
            print(score)
            
        return scores
