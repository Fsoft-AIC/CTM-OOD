import torch
from torch import nn
from torch.autograd import Variable
from .base_detector import BaseDetector


class ODINDetector(BaseDetector):
    def __init__(self, * , T, noise_magnitude, DATASET_STD, **kwargs) -> None:
        super().__init__('odin')
        self.T = T
        self.noise_magnitude=noise_magnitude
        self.require_grad = True
        self.DATASET_STD = DATASET_STD
    
    def score_batch(self, forward_fn, batch):
        batch[0] = batch[0].requires_grad_()
        logits = forward_fn(batch)
        preds = torch.argmax(logits.detach(), dim=1)
        odin_score = ODIN(batch[0], logits, forward_fn, self.T, self.noise_magnitude, self.DATASET_STD)
        
        # minus since lower score is more ID
        return {
            "scores": odin_score.detach(), 
            "preds": preds
        }
    
    def __str__(self) -> str:
        return f"{self.name}_T={self.T}_noise={self.noise_magnitude}"


def ODIN(inputs, logits, forward_fn, temper, noiseMagnitude1, DATASET_STD):
    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input
    criterion = nn.CrossEntropyLoss()

    # Using temperature scaling
    logits = logits / temper
    maxIndexTemp = torch.argmax(logits.detach(), dim=1)

    loss = criterion(logits, maxIndexTemp)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient =  torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2
    
    # Normalizing the gradient to the same space of image
    gradient[:,0] = (gradient[:,0] )/ DATASET_STD[0]
    gradient[:,1] = (gradient[:,1] )/ DATASET_STD[1]
    gradient[:,2] = (gradient[:,2] )/ DATASET_STD[2]

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data, gradient, alpha=-noiseMagnitude1)
    with torch.no_grad():
        logits = forward_fn((tempInputs,))
    logits = logits / temper
    # Calculating the confidence after adding perturbations
    probs = logits.softmax(dim=1)
    nnOutputs = torch.max(probs, dim=1).values
    return nnOutputs