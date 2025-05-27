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

class SCELoss(torch.nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, num_classes=6):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce = torch.nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)

        probs = torch.softmax(logits, dim=1)
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).float()
        targets_onehot = torch.clamp(targets_onehot, min=1e-4, max=1.0)  # evita log(0)

        rce_loss = (-probs * torch.log(targets_onehot)).sum(dim=1).mean()

        return self.alpha * ce_loss + self.beta * rce_loss
