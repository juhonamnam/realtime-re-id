import torch
import torch.nn as nn

__all__ = ['GateLoss']

class FocalLoss(nn.Module):
    """Implementation of Focal Loss for binary classification.

    Focal loss is designed to address class imbalance by down-weighting well-classified
    examples and focusing on hard examples.

    Attributes:
        alpha (float): Weighting factor for the classes.
        gamma (float): Focusing parameter for hard examples.
        epsilon (float): Small value for numerical stability.
    """
    def __init__(self, alpha=1, gamma=2, epsilon=1e-8):
        """Initializes FocalLoss.

        Args:
            alpha (float, optional): Alpha parameter. Defaults to 1.
            gamma (float, optional): Gamma parameter. Defaults to 2.
            epsilon (float, optional): Epsilon for stability. Defaults to 1e-8.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, predicted, target_mask):
        """Calculates Focal Loss between prediction and target.

        Args:
            predicted (torch.Tensor): Predicted probabilities.
            target_mask (torch.Tensor): Binary target mask.

        Returns:
            torch.Tensor: Loss tensor.
        """
        pt = torch.where(target_mask, predicted, 1 - predicted)
        pt = torch.clamp(pt, min=self.epsilon, max=1-self.epsilon)
        loss = - self.alpha * (1 - pt) ** self.gamma * torch.log(pt)
        return loss

class GateLoss(nn.Module):
    """Calculates Triplet Loss with hardest positive and hardest negative mining.

    This implementation is designed to handle multiple embedding vectors per item,
    weighting their contributions based on visibility scores.

    Attributes:
        margin (float): The margin between positive and negative distances.
        ranking_loss (nn.MarginRankingLoss): Underlying ranking loss module.
    """
    def __init__(self):
        """Initializes TripletLoss.

        Args:
            margin (float, optional): Margin for triplet loss. Defaults to 0.3.
            topk (int, optional): Number of top vectors to consider for loss calculation. Defaults to 3.
        """
        super(GateLoss, self).__init__()

    def forward(self, v_scores, emb_vecs, emb_vec_gates, id_labels):
        """Calculates the triplet loss.

        Args:
            v_scores (torch.Tensor): Visibility scores for each embedding of shape (Batch, VectorNum).
            emb_vecs (torch.Tensor): Embedding vectors of shape (Batch, VectorNum, VectorLen).
            emb_vec_gates (torch.Tensor): Gate values for each embedding of shape (Batch, VectorNum, VectorLen).
            id_labels (torch.Tensor): Identity labels for the batch of shape (Batch).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        v_scores = v_scores.detach()
        emb_vecs = emb_vecs.detach()
        batch, vector_num, vector_len = emb_vecs.shape

        v_score_matrix = torch.stack((v_scores.unsqueeze(0).expand(batch, -1, -1),
                                     v_scores.unsqueeze(1).expand(-1, batch, -1)), dim=-2) # batch x batch x 2
        v_score_matrix = (v_score_matrix.amin(dim=2, keepdim=True)
                          .unsqueeze(4).expand(-1, -1, 2, -1, vector_len))                 # batch x batch x 1 x vector_num x vector_len

        gate_matrix = torch.stack((emb_vec_gates.unsqueeze(0).expand(batch, -1, -1, -1),
                                  emb_vec_gates.unsqueeze(1).expand(-1, batch, -1, -1)),
                                  dim=2)                                                   # batch x batch x 2 x vector_num x vector_len

        eq_mask = id_labels.expand(batch, batch).eq(id_labels.expand(batch, batch).t())    # batch x batch
        eq_mask = (eq_mask
                   .unsqueeze(2).unsqueeze(3).unsqueeze(4)
                   .expand(-1, -1, 2, vector_num, vector_len)).float()                     # batch x batch x 2 x vector_num x vector_len

        vec_divert_mask = emb_vecs.unsqueeze(0) * emb_vecs.unsqueeze(1) > 0                # batch x batch x vector_num x vector_len
        vec_divert_mask = vec_divert_mask.unsqueeze(2).expand(-1, -1, 2, -1, -1)           # batch x batch x 2 x vector_num x vector_len

        fl = FocalLoss()

        weights = v_score_matrix * eq_mask

        f_loss = fl(gate_matrix, vec_divert_mask)
        f_loss = f_loss * weights 
        f_loss = torch.sum(f_loss) / batch

        return f_loss
