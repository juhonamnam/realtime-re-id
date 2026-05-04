import torch

__all__ = ['get_emb_similarity_vector',
           'prepare_for_similarity_metric']

def get_emb_similarity_vector(emb_vec1, emb_vec2):
    """Calculates the element-wise product of normalized embedding vectors.

    This represents the components of cosine similarity before summation.

    Args:
        emb_vec1 (torch.Tensor): First embedding vector.
        emb_vec2 (torch.Tensor): Second embedding vector.

    Returns:
        torch.Tensor: Element-wise product of L2-normalized vectors.
    """
    # Cosine Similarity
    norm1 = torch.norm(emb_vec1, p=2)
    norm2 = torch.norm(emb_vec2, p=2)

    unit_v1 = emb_vec1 / norm1
    unit_v2 = emb_vec2 / norm2

    product = unit_v1 * unit_v2

    return product

def get_emb_similarity_score(emb_vec1, emb_vec2):
    """Calculates the cosine similarity score between two embedding vectors.

    Args:
        emb_vec1 (torch.Tensor): First embedding vector.
        emb_vec2 (torch.Tensor): Second embedding vector.

    Returns:
        torch.Tensor: Cosine similarity score.
    """
    # Cosine Similarity
    norm1 = torch.norm(emb_vec1, p=2)
    norm2 = torch.norm(emb_vec2, p=2)

    unit_v1 = emb_vec1 / norm1
    unit_v2 = emb_vec2 / norm2

    score = torch.mm(torch.unsqueeze(unit_v1, 0), torch.unsqueeze(unit_v2, 0).T).squeeze()
    
    return score

def prepare_for_similarity_metric(feature1, feature2):
    """Prepares features from two images for metric calculation.

    Calculates individual part similarity scores and combined visibility scores.

    Args:
        feature1 (tuple): (v_scores, emb_vecs) for the first image.
        feature2 (tuple): (v_scores, emb_vecs) for the second image.

    Returns:
        tuple: (v_scores, part_s_scores)
            v_scores (torch.Tensor): Minimum visibility score for each part across both images.
            part_s_scores (list[torch.Tensor]): Similarity score for each part.
    """
    ft1_v_scores, ft1_emb_vecs = feature1
    ft2_v_scores, ft2_emb_vecs = feature2

    part_s_scores = []

    for emb_vec1, emb_vec2 in zip(ft1_emb_vecs, ft2_emb_vecs):
        s_score = get_emb_similarity_score(emb_vec1, emb_vec2)

        if s_score.isnan():
            s_score = torch.tensor(-1.).view(s_score.shape).to(s_score.device)

        part_s_scores.append(s_score)
    
    v_scores = torch.cat((ft1_v_scores.unsqueeze(0), ft2_v_scores.unsqueeze(0)), dim=0).min(dim=0).values

    return v_scores, part_s_scores