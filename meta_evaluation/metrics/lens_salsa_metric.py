from lens import download_model
from lens import LENS_SALSA

class LENS_SALSA_metric:

    name = "LENS_SALSA"

    def __init__(self):
        self.lens_metric_salsa = LENS_SALSA(download_model("davidheineman/lens-salsa"))

    def compute_metric(self, complex, simplified, references):

        scores, _ = self.lens_metric_salsa.score([c.lower() for c in complex], [s.lower() for s in simplified], 
                                    batch_size=16, devices=[0])

        print(scores)
        return scores