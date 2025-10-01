import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def InfoNCELoss(logits):
    '''
    logits: 1 positive + n negative per batch
    '''
    b = logits.shape[0]
    n = logits.shape[1] - 1
    device = logits.device
    labels = torch.zeros((b)).long().to(device)

    loss = F.cross_entropy(logits, labels)

    result = F.softmax(logits, dim=1)
    print(loss.mean(), result[0])

    return loss.mean()
