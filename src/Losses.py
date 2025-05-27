import torch
import torch.nn.functional as F

class GCODLoss(torch.nn.Module):
    def __init__(self, gamma=0.2):
        super(GCODLoss, self).__init__()
        self.gamma = gamma  # hyperparametro che controlla il peso della parte robusta

    def forward(self, logits, labels):
        # Cross Entropy standard
        ce_loss = F.cross_entropy(logits, labels, reduction='none')  # (batch_size,)

        # Compute probabilities
        probs = F.softmax(logits, dim=1)
        true_probs = probs[range(len(labels)), labels]  # Probabilità del target corretto

        # Peso GCOD: bassa confidenza → peso basso
        weight = (true_probs.detach() ** self.gamma)

        loss = weight * ce_loss
        return loss.mean()
