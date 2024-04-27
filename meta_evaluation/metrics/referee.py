import torch

from models.deberta_for_eval import DebertaForEval

DERBERTA_MODEL_DIR = "/Users/mmaddela3/Documents/simplification_evaluation/external_models/pretrained/checkpoints/deberta/"
DERBERTA_TOKENIZER_DIR = "/Users/mmaddela3/Documents/simplification_evaluation/external_models/pretrained/tokenizers/deberta"
CHECKPOINTS_DIR = "/Users/mmaddela3/Documents/simplification_evaluation/external_models/overall_ratings_checkpoints"

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
            print(score)
        return scores
