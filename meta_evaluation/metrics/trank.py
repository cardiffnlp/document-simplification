import torch
import torch.nn as nn
from transformers import AutoModel
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoTokenizer

# Define the model:
BASE_MODEL = "Peltarion/xlm-roberta-longformer-base-4096"
class ReadabilityModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, model_name=BASE_MODEL):
        super(ReadabilityModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(768, 1)

    def forward(self, ids, mask):
        out = self.model(input_ids=ids, attention_mask=mask,
                         output_hidden_states=False)
        out = self.drop(out[1])
        outputs = self.fc(out)

        return outputs

class TRank:

    name = "TRank"

    def __init__(self):
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the model:
        self.model = ReadabilityModel.from_pretrained("trokhymovych/TRank_readability")
        
        # Load the tokenizer:
        self.tokenizer = AutoTokenizer.from_pretrained("trokhymovych/TRank_readability")

        self.model.eval()


    def compute_metric(self, complex, simplified, references):
        scores = []
        
        for input_text in simplified:
            # Tokenize the input text
            inputs = self.tokenizer.encode_plus(
                input_text,
                add_special_tokens=True,
                max_length=512,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            ids = inputs['input_ids']
            mask = inputs['attention_mask']
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(ids, mask)
                readability_score = outputs.item()
                
            # Join the readability scores
            
            scores.append(readability_score)

        return scores
