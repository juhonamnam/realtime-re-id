import torch


def get_part_distances(emb_vecs1, emb_vecs2):
    part_distances = (emb_vecs1 - emb_vecs2).norm(p=2, dim=1)
    return part_distances


def prepare_for_distance_metric(feature1: tuple, feature2: tuple) -> tuple:
    ft1_v_scores, ft1_emb_vecs = feature1
    ft2_v_scores, ft2_emb_vecs = feature2

    part_distances = get_part_distances(ft1_emb_vecs, ft2_emb_vecs)

    v_scores = torch.cat((ft1_v_scores.unsqueeze(
        0), ft2_v_scores.unsqueeze(0)), dim=0).amin(dim=0)

    return v_scores, part_distances


def prepare_for_distance_metric_cross_batch(feature1: tuple, feature2: tuple) -> tuple:
    ft1_v_scores, ft1_emb_vecs = feature1
    ft2_v_scores, ft2_emb_vecs = feature2

    part_distances = torch.cdist(ft1_emb_vecs.permute(1, 0, 2),
                                 ft2_emb_vecs.permute(1, 0, 2), p=2)

    v_scores = torch.min(ft1_v_scores.permute(1, 0).unsqueeze(2),
                         ft2_v_scores.permute(1, 0).unsqueeze(1))

    return v_scores, part_distances
