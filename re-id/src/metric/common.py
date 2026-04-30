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
