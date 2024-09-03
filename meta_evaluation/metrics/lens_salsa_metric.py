from lens import download_model
from lens import LENS_SALSA

class LENS_SALSA_metric:

    name = "LENS_SALSA"

    def __init__(self):
        self.lens_metric_salsa = LENS_SALSA(download_model("davidheineman/lens-salsa"))

    def compute_metric(self, complex, simplified, references):

        all_scores = []
        for comp, simp in zip(complex, simplified):
        
            scores, _ = self.lens_metric_salsa.score([comp.lower()], [simp.lower()], 
                                        batch_size=1, devices=[0])
            all_scores.append(scores[0])

        return all_scores