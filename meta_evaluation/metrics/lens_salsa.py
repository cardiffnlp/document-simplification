from lens import download_model
from lens.lens_salsa import LENS_SALSA

class LENS_SALSA:

    name = "LENS_SALSA"

    def __init__(self, model_path):
        model_path = download_model("davidheineman/lens-salsa") 
        self.lens_metric = LENS_SALSA(model_path, rescale=True)

    def compute_metric(self, complex, simplified, references):
        scores = self.lens_metric.score(complex, 
                                        simplified, 
                                    batch_size=1, gpus=0)
        
        assert len(scores) == len(complex) == len(simplified) == len(new_references)
        return scores