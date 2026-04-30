import torch

def get_emb_similarity_vector(emb_vec1, emb_vec2):
    # Cosine Similarity
    norm1 = torch.norm(emb_vec1, p=2)
    norm2 = torch.norm(emb_vec2, p=2)

    unit_v1 = emb_vec1 / norm1
    unit_v2 = emb_vec2 / norm2

    product = unit_v1 * unit_v2

    return product

def get_emb_similarity_score(emb_vec1, emb_vec2):
    # Cosine Similarity
    norm1 = torch.norm(emb_vec1, p=2)
    norm2 = torch.norm(emb_vec2, p=2)

    unit_v1 = emb_vec1 / norm1
    unit_v2 = emb_vec2 / norm2

    score = torch.mm(torch.unsqueeze(unit_v1, 0), torch.unsqueeze(unit_v2, 0).T).squeeze()
    
    return score

def prepare_for_similarity_metric(feature1, feature2):
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