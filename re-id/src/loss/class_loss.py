import torch.nn as nn

class ClassLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, class_logits, id_labels):
        CE = nn.CrossEntropyLoss()

        class_loss = CE(class_logits, id_labels)

        return class_loss
