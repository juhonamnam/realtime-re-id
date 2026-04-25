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

VALUE_AT_THRESHOLD = 0.7

def get_similarity_score(feature1, feature2,
                         thresholds=0.5,
                         exp=2, return_part_scores=False):
    ft1_v_scores, ft1_emb_vecs = feature1
    ft2_v_scores, ft2_emb_vecs = feature2

    ft_len = ft1_emb_vecs.shape[0]

    if not isinstance(thresholds, list):
        thresholds = [thresholds] * ft_len

    combined_v_scores = torch.cat((ft1_v_scores.unsqueeze(0), ft2_v_scores.unsqueeze(0)), dim=0).min(dim=0).values

    total_s_score = torch.tensor(1.).to(combined_v_scores.device)

    if return_part_scores:
        part_s_scores = []

    for i in range(ft_len):
        v_score = combined_v_scores[i]
        unknown_ratio = 1 - v_score

        emb_vec1 = ft1_emb_vecs[i]
        emb_vec2 = ft2_emb_vecs[i]
        s_score = get_emb_similarity_score(emb_vec1, emb_vec2)

        if s_score.isnan():
            s_score = torch.tensor(-1.).view(s_score.shape).to(s_score.device)

        if return_part_scores:
            part_s_scores.append(s_score)
        
        threshold = (thresholds[i] + 1) / 2
        s_score = (s_score + 1) / 2

        if s_score < threshold:
            s_score = VALUE_AT_THRESHOLD / (threshold ** exp) * (s_score ** exp)
        else:
            s_score = 1 - ((1 - VALUE_AT_THRESHOLD) / ((1 - threshold) ** exp)) * ((1 - s_score) ** exp)

        total_s_score *= unknown_ratio + v_score * s_score

    if return_part_scores:
        return total_s_score, part_s_scores

    return total_s_score
