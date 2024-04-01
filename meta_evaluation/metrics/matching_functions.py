from metrics.bleu import BLEU
from metrics.sari import SARI

# class MatchingFunction(metaclass=abc.ABCMeta):
#   """Interface for matching function APIs."""

#   @abc.abstractmethod
#   def __call__(
#       self,
#       reference,
#       candidate,
#   ):
#     raise NotImplementedError()
  

class BleuMatchingFunction:
  
  def __init__(self):
    self.metric = BLEU()

  def __call__(
      self,
      source,
      reference,
      candidate,
  ):
    if isinstance(reference, list):
      return [
          self.metric.compute_metric("", c, [r]) for r, c in zip(reference, candidate)]
    else:
      return self.metric.compute_metric("", candidate, [reference])
    

class SARIMatchingFunction:
  
  def __init__(self):
    self.metric = SARI()

  def __call__(
      self,
      source,
      reference,
      candidate,
  ):
    if isinstance(reference, list):
      return [
          self.metric.compute_metric(s, c, [r]) for s, r, c in zip(source, reference, candidate)]
    else:
      return self.metric.compute_metric(source, candidate, [reference])

