import torch


# Load the necessary model and checkpoints from https://github.com/i-need-sleep/referee/tree/main?tab=readme-ov-file
DERBERTA_MODEL_DIR = "/home/ubuntu/simplification/external/referee/pretrained/checkpoints/deberta"
DERBERTA_TOKENIZER_DIR = "/home/ubuntu/simplification/external/referee/pretrained/tokenizers/deberta"
CHECKPOINTS_DIR = "/home/ubuntu/simplification/external/referee/pretrained"


import torch
from transformers import AutoModel, AutoTokenizer

# Code copied from REFEREE paper: https://github.com/i-need-sleep/referee/blob/main/code/models/deberta_for_eval.py
class DebertaForEval(torch.nn.Module):
    def __init__(self, model_path, tokenizer_path, device, n_supervision=13, head_type='mlp', backbone='deberta'):
        super(DebertaForEval, self).__init__()
        self.n_supervision = n_supervision
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.deberta = AutoModel.from_pretrained(model_path)
        self.backbone = backbone

        if backbone == 'deberta':
            self.hidden_size = 768
        elif backbone == 'roberta':
            self.hidden_size = 1024
        else:
            raise NotImplementedError

        self.head_type = head_type
        if head_type == 'mlp':
            self.regression_heads_layer_1 = torch.nn.ModuleList([torch.nn.Linear(self.hidden_size, 512) for i in range(n_supervision)])
            self.regression_heads_layer_2 = torch.nn.ModuleList([torch.nn.Linear(512, 1) for i in range(n_supervision)])
            self.relu = torch.nn.ReLU()
        elif head_type == 'linear':
            self.linear_out = torch.nn.ModuleList([torch.nn.Linear(self.hidden_size, 1) for i in range(n_supervision)])
        else:
            raise NotImplementedError

        self.to(device)
        self.float()
    
    def forward(self, sents):
        if self.backbone == 'deberta':
            tokenized = self.tokenizer(sents, padding=True, truncation=True, max_length=512)
            input_ids = torch.tensor(tokenized['input_ids']).to(self.device)
            token_type_ids =  torch.tensor(tokenized['token_type_ids']).to(self.device)
            attention_mask =  torch.tensor(tokenized['attention_mask']).to(self.device)
            model_out = self.deberta(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0][:, 0, :] # Take the emb for the first token
        elif self.backbone == 'roberta':
            encoded_input = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True, max_length=512)
            for key, val in encoded_input.items():
                encoded_input[key] = val.to(self.device)
            model_out = self.deberta(**encoded_input)[0][:, 0, :]
        else:
            raise NotImplementedError

        heads_out = []
        for head_idx in range(self.n_supervision):
            if self.head_type == 'mlp':
                head_out = self.regression_heads_layer_1[head_idx](model_out)
                head_out = self.relu(head_out)
                head_out = self.regression_heads_layer_2[head_idx](head_out)
                heads_out.append(head_out)
            elif self.head_type == 'linear':
                head_out = self.linear_out[head_idx](model_out)
                heads_out.append(head_out)

        heads_out = torch.cat(heads_out, dim=1)
        return heads_out # [batch_size, n_head]

class REFEREE:

    name = "REFEREE"

    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DebertaForEval(DERBERTA_MODEL_DIR, DERBERTA_TOKENIZER_DIR, 
                                    device, head_type='linear')
        self.model.load_state_dict(torch.load(f'{CHECKPOINTS_DIR}/pretrained_deberta.bin',
                                    map_location=device)['model_state_dict'], strict=False)
        self.model.eval()


    def compute_metric(self, complex, simplified, references):
        scores = []
        for single_comp, single_simp in zip(complex, simplified):
            sep_token = self.model.tokenizer.sep_token
            model_input = [single_comp + ' ' + sep_token + ' ' + single_simp]
            model_out = self.model(model_input)
            score = model_out[:, -1].item()
            scores.append(score)
        return scores
