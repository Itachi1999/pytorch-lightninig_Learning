import torch
from torchmetrics import Metric

# My Metric (Mostly Accuracy)
class myAccuracy(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state("correct", default = torch.tensor(0), dist_reduce_fx = "sum")
        self.add_state("samples", default = torch.tensor(0), dist_reduce_fx = "sum")
    
    def update(self, scores, target):
        preds = torch.argmax(scores, dim = 1)
        assert preds.shape == target.shape
        self.correct += (preds == target).sum()
        self.samples += target.numel()

    def compute(self):
        return self.correct / self.samples
