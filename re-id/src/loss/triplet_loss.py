import torch
import torch.nn as nn

__all__ = ['TripletLoss']

class TripletLoss(nn.Module):
    """Calculates Triplet Loss with hardest positive and hardest negative mining.

    This implementation is designed to handle multiple embedding vectors per item,
    weighting their contributions based on visibility scores.

    Attributes:
        margin (float): The margin between positive and negative distances.
        ranking_loss (nn.MarginRankingLoss): Underlying ranking loss module.
    """
    def __init__(self, margin=0.3, topk=1):
        """Initializes TripletLoss.

        Args:
            margin (float, optional): Margin for triplet loss. Defaults to 0.3.
            topk (int, optional): Number of top vectors to consider for loss calculation. Defaults to 3.
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.topk = topk
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, vectors, targets):
        """Calculates the triplet loss.

        Args:
            vectors (torch.Tensor): Embedding vectors of shape (Batch, VectorNum, VectorLen).
            targets (torch.Tensor): Identity labels for the batch of shape (Batch).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        batch = vectors.size(0)

        # # Part-wise L2 Distance
        # vectors = vectors.permute(1, 0, 2)                                             # vector_num x batch x vector_len
        # dist = torch.cdist(vectors, vectors, p=2)                                      # vector_num x batch x batch
        # dist = dist.mean(dim=0)                                                        # batch x batch

        # Concat L2 Distance
        vectors = vectors.flatten(1, 2)                                                  # batch x (vector_num * vector_len)
        dist = torch.cdist(vectors, vectors, p=2)                                        # batch x batch

        mask = targets.expand(batch, batch).eq(targets.expand(batch, batch).t())       # batch x batch

        dist_ap = dist.masked_fill(~mask, -1).topk(self.topk, dim=1)                   # hardest positives
        dist_an = dist.masked_fill(mask, 1e12).topk(self.topk, dim=1, largest=False)   # hardest negatives

        dist_ap = dist_ap.values.flatten()
        dist_an = dist_an.values.flatten()

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss
