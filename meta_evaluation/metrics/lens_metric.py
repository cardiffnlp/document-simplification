from lens import LENS, download_model


class LENS_metric:

    name = "LENS"

    def __init__(self):
        self.lens_metric = LENS(download_model("davidheineman/lens"), rescale=True)

    def compute_metric(self, complex, simplified, references):
        new_references = []
        for refs in references:
            new_references.append([ref.lower() for ref in refs])

        scores = self.lens_metric.score([c.lower() for c in complex], 
                                        [s.lower() for s in simplified],
                                        new_references,  
                                    batch_size=16, devices=[0])
        
        assert len(scores) == len(complex) == len(simplified) == len(new_references)
        return scores
