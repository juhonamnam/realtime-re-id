import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, margin=0.6):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, weights, vectors, targets):
        """
        Args:
            vectors: batch x vector_num x vector_len
            weights: batch x vector_num
            targets: batch
        """
        weights = weights.detach()
        batch, vector_num, _ = vectors.shape

        # Cosine Distance
        unit_vectors = vectors / vectors.norm(p=2,
                                              dim=-1,
                                              keepdim=True).clamp(min=1e-12)           # batch x vector_num x vector_len
        cosine_similarity = torch.einsum("abf,cbf->bca", unit_vectors, unit_vectors)   # vector_num x batch x batch
        dist = 1 - cosine_similarity                                                   # vector_num x batch x batch

        dist_w = weights.t().unsqueeze(1).expand(vector_num, batch, batch)             # vector_num x batch x batch
        dist_w = torch.min(dist_w, dist_w.permute(0, 2, 1))                            # vector_num x batch x batch
        dist_w /= dist_w.sum(dim=0, keepdim=True).clamp(min=1e-12)                     # vector_num x batch x batch

        dist = (dist * dist_w).sum(dim=0)                                              # batch x batch

        mask = targets.expand(batch, batch).eq(targets.expand(batch, batch).t())       # batch x batch

        dist_ap = dist.masked_fill(~mask, -1).amax(dim=1)  # hardest positive
        dist_an = dist.masked_fill(mask, 1e12).amin(dim=1) # hardest negative

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss
