from lens.lens_score import LENS


class LENS_metric:

    name = "LENS"

    def __init__(self, model_path):
        self.lens_metric = LENS(model_path, rescale=True)

    def compute_metric(self, complex, simplified, references):
        scores = self.lens_metric.score([complex.lower()], 
                                        [simplified.lower()],
                                    [[ref.lower() for ref in references]],  
                                    batch_size=1, gpus=0)
        return scores[0]
