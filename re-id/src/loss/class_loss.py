import torch.nn as nn

class ClassLoss(nn.Module):
    """Calculates cross-entropy loss for identity classification.

    This loss is used during the training of the Re-ID model to ensure that the
    extracted features are discriminative for different identities.
    """
    def __init__(self):
        """Initializes ClassLoss."""
        super().__init__()

    def forward(self, class_logits, id_labels):
        """Calculates the cross-entropy loss.

        Args:
            class_logits (torch.Tensor): Predicted logits of shape (Batch, NumClasses).
            id_labels (torch.Tensor): Ground truth labels of shape (Batch).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        CE = nn.CrossEntropyLoss()

        class_loss = CE(class_logits, id_labels)

        return class_loss
