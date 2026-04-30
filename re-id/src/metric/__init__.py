import torch.nn as nn

from .common import *
from .weakest_link import get_weakest_link_similarity_score
from .weighted_avg import get_weighted_avg_similarity_score

class SimilarityMetric(nn.Module):
    def __init__(self, method='weakest_link', thresholds=0.5):
        super(SimilarityMetric, self).__init__()
        if method == 'weakest_link':
            self.similarity_func = get_weakest_link_similarity_score
        elif method == 'weighted_avg':
            self.similarity_func = get_weighted_avg_similarity_score
        else:
            raise ValueError(f"Unsupported similarity metric method: {method}")
        self.thresholds = thresholds

    def forward(self, feature1, feature2, thresholds=None, return_part_scores=False):
        if thresholds is None:
            thresholds = self.thresholds
        return self.similarity_func(feature1, feature2, thresholds, return_part_scores)